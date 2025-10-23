import ast
from .base import BaseStrategy
from typing import Sequence
from typing_extensions import TypedDict
from typing_extensions import Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import trim_messages
from langchain.tools.base import StructuredTool
from langgraph.graph import END, StateGraph


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query_result: str
    user_country: str


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


class CouponChain(BaseStrategy):
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

        if max_tokens > 0:
            self.trimmer = trim_messages(
                max_tokens=max_tokens,
                strategy="last",
                token_counter=llm,
                include_system=True,
                allow_partial=False,
                start_on="human",
            )

        self.graph = self._build_graph(checkpointer=self.checkpointer)

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
        graph_builder.add_node("guard_rails", self._apply_guard_rails)

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
        graph_builder.add_edge("generate_answer", "guard_rails")
        graph_builder.add_edge("guard_rails", END)

        return graph_builder.compile(checkpointer=checkpointer)

    def query_or_respond(self, state: State):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = self.llm.bind_tools(
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

    def conditional_tool_calling(self, state: State):
        # Check if the latest message is a tool call
        ai_message = state["messages"][-1]
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return ai_message.tool_calls[0]["name"]

        return END

    def stream(self, input, config=None, **kwargs):
        user_country = "AE" if "user_country" not in kwargs else kwargs["user_country"]
        return self.graph.stream(
            {
                "messages": [{"role": "user", "content": input}],
                "user_country": user_country,
            },
            config,
            **kwargs,
        )
