import ast
from config import Config
from typing import Sequence
from typing_extensions import TypedDict
from typing_extensions import Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_toolkits import create_retriever_tool
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import trim_messages
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain.tools.base import StructuredTool


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query_result: str
    user_country: str


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


class CouponChain:
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

        if max_tokens > 0:
            self.trimmer = trim_messages(
                max_tokens=max_tokens,
                strategy="last",
                token_counter=llm,
                include_system=True,
                allow_partial=False,
                start_on="human",
            )

        self.checkpointer = checkpointer
        if checkpointer is not None:
            checkpointer.setup()

        self.graph = self._build_graph(checkpointer=self.checkpointer)

    def _create_category_retriever(self, category_names, score_threshold=0.5, k=1):
        _ = self.vector_store.add_texts(category_names)

        return self.vector_store.as_retriever(
            # search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score_threshold, "k": k},
        )

    def _build_graph(self, checkpointer=None):
        graph_builder = StateGraph(state_schema=State)

        graph_builder.add_node(
            "find_coupons", ToolNode([StructuredTool.from_function(self.find_coupons)])
        )
        graph_builder.add_node(
            "search_categories", ToolNode([self.search_categories_tool])
        )
        graph_builder.add_node(self.execute_query)
        graph_builder.add_node(self.query_or_respond)
        graph_builder.add_node(self.generate_answer)

        # graph_builder.add_edge(START, "query_or_respond")
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            self.conditional_tool_calling,
            [
                "find_coupons",
                "search_categories",
                END,
            ],
        )

        graph_builder.add_edge("find_coupons", "execute_query")
        graph_builder.add_edge(
            "search_categories", "query_or_respond"
        )  # query_or_respond will see the category from the conversation history
        graph_builder.add_edge("execute_query", "generate_answer")
        graph_builder.add_edge("execute_query", END)

        return graph_builder.compile(checkpointer=checkpointer)

    def query_or_respond(self, state: State):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools(
            [self.find_coupons, self.search_categories_tool]
        )

        system_msg = SystemMessage(
            content=(
                "You are a coupon chat bot. You have 1 job only: to assist users to find discount coupons."
                "Users can provide you with the name of the shop or a category of coupons they're looking for."
                f"If the user inputs what could be a category, search for coupons from websites with a category that is most similar to the one that the user provied. The {self.CATEGORY_LOOKUP_TOOL_NAME} can find the most similar category"
                "If the user inputs what could be a company, search for coupons for the given company or . Be fliexible when considering the user input as a company"
                # "If the user inputs a question without a company or category, let them know that you are a chatbot that can help find coupons given a company name or category."
                f"IMPORTANT: The user country is {state['user_country'] or None}"
            )
        )

        trimmed_messages = state["messages"]
        if self.trimmer is not None:
            trimmed_messages = self.trimmer.invoke(state["messages"])

        response = llm_with_tools.invoke([system_msg] + trimmed_messages)

        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}

    # @tool
    def find_coupons(self, query: str, country: str, category: str = None):
        """Generate SQL query to fetch information."""

        system_message = """
        Given an input question, create a syntactically correct {dialect} query to
        run to help find coupons relevant to the input question. Unless the user specifies in his question a
        specific number of examples they wish to obtain, always limit your query to
        at most {top_k} results. You can order the results by a relevant column to
        return the most interesting examples in the database.

        Never query for all the columns from a specific table, only ask for a the
        few relevant columns given the question.


        IMPORTANT: 
        - Pay attention to use only the column names that you can see in the schema
        description. Be careful to not query for columns that do not exist. Also,
        pay attention to which column is in which table.
        - your query must handle casing
        - The coupon must either not have valid countries, or it must be valid {country}
        - The coupon must either not have an expiry date, or the expiry date is in the future.
        - Focus on coupons for category {category}

        Only use the following tables:
        {table_info}
        """

        user_prompt = "Question: {input}"
        query_prompt_template = ChatPromptTemplate(
            [("system", system_message), ("user", user_prompt)]
        )

        prompt = query_prompt_template.invoke(
            {
                "dialect": self.coupon_db.dialect,
                "top_k": 10,
                "table_info": self.coupon_db.get_table_info(
                    table_names=self.allowed_tables
                ),
                "input": query,
                "country": country,
                "category": category if category is not None else "any category",
            }
        )
        structured_llm = self.llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        return result["query"]

    def execute_query(self, state: State):
        """Execute SQL query."""

        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        execute_query_tool = QuerySQLDatabaseTool(db=self.coupon_db)
        result = execute_query_tool.invoke(tool_messages[0].content)
        return {"query_result": result}

    def generate_answer(self, state: State):
        """Answer question using retrieved information as context."""

        question = None
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                question = message.content
                break

        if not question:
            question = "User question not found"

        prompt = (
            "Given the following user question and matching coupons, "
            "List the shop name, coupon code, expiry date (if available) and any terms and conditions. the content must be in plain text, suitable for a WhatsApp message.\n\n",
            "If no coupons are found, apologise and recommend a different search."
            f"Question: {question}\n"
            f"Matching coupons: {state['query_result']}",
        )
        response = self.llm.invoke(prompt)
        return {"messages": [response.content]}

    def conditional_tool_calling(seld, state: State):
        # Check if the latest message is a tool call
        ai_message = state["messages"][-1]
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return ai_message.tool_calls[0]["name"]

        return END

    def converse(self):
        from uuid import uuid4

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

            for step in self.graph.stream(
                {
                    "messages": [{"role": "user", "content": question}],
                    "user_country": "AE",
                },
                stream_mode="values",
                config=config,
            ):
                step["messages"][-1].pretty_print()


def create_category_retriever(db: SQLDatabase, vector_store, score_threshold=0.5, k=1):
    res = db.run("SELECT name FROM categories")
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    categories = list(set(res))

    if categories is not None and len(categories) > 0:
        _ = vector_store.add_texts(categories)

        return vector_store.as_retriever(
            # search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score_threshold, "k": k},
        )
    return None


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
        coupon_chatbot = CouponChain(
            llm=llm,
            vector_store=vector_store,
            checkpointer=checkpointer,
            cache=InMemoryCache(),
            max_tokens=384,
            coupon_db=coupon_db,
            allowed_tables=allowed_tables,
            category_names=categories,
        )

        coupon_chatbot.converse()
