import ast
from config import Config
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_toolkits import create_retriever_tool
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from uuid import uuid4
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache


class CouponAgent:
    CATEGORY_LOOKUP_TOOL_NAME = "search_categories"

    def __init__(
        self,
        llm,
        vector_store,
        coupon_db: SQLDatabase,
        allowed_tables=None,
        cache=None,
        category_names=[],
        max_tokens=0,
        checkpointer=None,
    ):
        assert llm is not None, "llm is required"
        assert coupon_db is not None, "coupon_db is required"
        assert vector_store is not None, "vector_store is required"
        assert (
            category_names is not None and len(category_names) > 0
        ), "category names are required"

        self.llm = llm
        self.coupon_db = coupon_db
        self.vector_store = vector_store

        if cache is not None:
            set_llm_cache(cache)

        self.allowed_tables = allowed_tables
        if allowed_tables is not None:
            allowed_tables = coupon_db.get_usable_table_names()

        self.search_categories_tool = create_retriever_tool(
            self._create_category_retriever(category_names=category_names),
            name=self.CATEGORY_LOOKUP_TOOL_NAME,
            description=(
                "Use to look up categories to filter on. Input is an approximate spelling "
                "of the category, output is known category. Use the cateogry most similar to the search."
            ),
        )

        self.max_tokens = max_tokens
        self.checkpointer = checkpointer
        if checkpointer is not None:
            checkpointer.setup()

        system_message = """
            You are an agent designed to retrieve coupons for a particular store or for a category from a SQL database.
            Given an input store name or category, create a syntactically correct {dialect} query to run,
            then look at the results of the query and return the answer. Unless the user
            specifies a specific number of examples they wish to obtain, always limit your
            query to at most {top_k} results.

            - You can order the results by a relevant column to return the most interesting
            examples in the database. 

            - Never query for all the columns from a specific table,only ask for the relevant columns given the question.

            - You must return the coupon code, shop name, expiry date, summarize the terms and conditions.

            - You MUST double check your query before executing it. If you get an error while
            executing a query, rewrite the query and try again.

            - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
            database.

            - IMPORTANT: Only consider the following tables {allowed_tables}

            - If the user inputs what could be a company or a category, search for coupons for the company. Be fliexible when considering the user input as a company or category.
            - If the user inputs a question without a company or category, let them know that you are a chatbot that can help find coupons given a company name or category.
            - Search for coupons for the user's country and for coupons where the country is not specified.
            - If no coupon is find, apologise and suggest other things to search for
            - If necessary, you can find similar categories using the {category_retriever} tool

            - Ensure all messages are user-friendly and cheerful!
        """.format(
            dialect=coupon_db.dialect,
            top_k=5,
            allowed_tables=allowed_tables,
            category_retriever=self.CATEGORY_LOOKUP_TOOL_NAME,
        )

        toolkit = SQLDatabaseToolkit(db=coupon_db, llm=llm)
        tools = toolkit.get_tools()
        tools.append(self.search_categories_tool)

        self.agent_executor = create_react_agent(
            llm,
            tools,
            prompt=system_message,
            checkpointer=checkpointer,
            pre_model_hook=self._pre_model_hook,
        )

    def _create_category_retriever(self, category_names, score_threshold=0.5, k=1):
        _ = self.vector_store.add_texts(category_names)

        return self.vector_store.as_retriever(
            # search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score_threshold, "k": k},
        )

    def _pre_model_hook(self, state):
        trimmed_messages = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=self.max_tokens,
            start_on="human",
            end_on=("human", "tool"),
        )
        # You can return updated messages either under `llm_input_messages` or
        # `messages` key (see the note below)
        # This will pass the trimmed messages to the LLM, but the message history in the graph state does not change.
        # In order to also trim the messages in the graph state
        # from langchain_core.messages import RemoveMessage
        # from langgraph.graph.message import REMOVE_ALL_MESSAGES
        # return {"llm_input_messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed_messages]}
        return {"llm_input_messages": trimmed_messages}

    def converse(self):
        thread_id = None
        while True:
            if thread_id is None:
                thread_id = str(uuid4())
                print(f"Thread Id: {thread_id}")

            config = {"configurable": {"thread_id": thread_id}}

            question = input("Enter your question (or type 'exit' to quit): ")
            if question.lower().strip() in ("exit", "quit", "q"):
                break

            if question.lower().startswith("reset"):
                thread_id = None
                parts = question.split(" ")
                if len(parts) > 1:
                    thread_id = parts[1]
                continue

            user_prompt = "`{input}`. The user's country is {user_country}"
            query_prompt_template = ChatPromptTemplate([("human", user_prompt)])
            prompt_value = query_prompt_template.invoke(
                {"input": question, "user_country": "AE"},
            )
            messages = prompt_value.to_messages()

            for step in self.agent_executor.stream(
                {"messages": messages}, stream_mode="values", config=config
            ):
                step["messages"][-1].pretty_print()


if __name__ == "__main__":
    config = Config()
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large", api_key=config.OPENAI_API_KEY
    )
    vector_store = InMemoryVectorStore(embeddings)
    llm = init_chat_model(
        "gpt-4o-mini", model_provider="openai", api_key=config.OPENAI_API_KEY
    )

    coupon_db = SQLDatabase.from_uri(config.COUPON_DB_CONN_STRING)
    # Todo, this should come from a config
    allowed_tables = ["categories", "country_list", "coupons", "websites"]

    res = coupon_db.run("SELECT name FROM categories")
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    categories = list(set(res))

    with PostgresSaver.from_conn_string(
        config.CHECKPOINT_DB_CONN_STRING
    ) as checkpointer:
        checkpointer.setup()
        coupon_chatbot = CouponAgent(
            llm=llm,
            cache=InMemoryCache(),
            max_tokens=384,
            checkpointer=checkpointer,
            coupon_db=coupon_db,
            allowed_tables=allowed_tables,
            category_names=categories,
        )

        coupon_chatbot.converse()
