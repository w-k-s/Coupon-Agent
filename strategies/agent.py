from .base import BaseStrategy
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.messages.utils import trim_messages, count_tokens_approximately


class CouponAgent(BaseStrategy):
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
        super().__init__(
            llm=llm,
            vector_store=vector_store,
            coupon_db=coupon_db,
            allowed_tables=allowed_tables,
            cache=cache,
            category_names=category_names,
            max_tokens=max_tokens,
            checkpointer=checkpointer,
        )

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
            pre_model_hook=self._pre_model_hook,  # This causes duplicate printing of messages
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

    def stream(self, question, config=None, **kwargs):

        user_country = "AE" if "user_country" not in kwargs else kwargs["user_country"]
        user_prompt = "`{input}`. The user's country is {user_country}"
        query_prompt_template = ChatPromptTemplate([("human", user_prompt)])
        prompt_value = query_prompt_template.invoke(
            {"input": question, "user_country": user_country},
        )
        messages = prompt_value.to_messages()
        return self.agent_executor.stream({"messages": messages}, config, **kwargs)
