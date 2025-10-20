from .base import BaseStrategy
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from guardrails.hub import DetectPII
from guardrails import Guard
from langchain_core.messages.ai import AIMessage


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
            You are an agent designed to retrieve coupons from a SQL database.

            You will receive a question from the user. 

            You must identify if the question is the name of a shop, a category, a coupon-related query, or commentary

            Given a shop name, create a syntactically correct {dialect} query to run,
            then look at the results of the query and return the answer.
            If no coupon is find, apologise and ask the user to try a different question.

            Given a category, you may use the {category_retriever} tool to find the closest available category.
            You must then create a syntactically correct {dialect} query to find coupons for the given category.

            Given a coupon-related query, try to assist the user as much as possible without breaking any rules.
            
            Given commentary, engage the user as much as possible without breaking rules. Steer the conversation to finding coupons

            Rules:
            
            1. Unless the user specifies a specific number of examples they wish to obtain, always limit your
            query to at most {top_k} results.

            2. IMPORTANT: Only consider the following tables {allowed_tables}.

            3. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
            database.

            4. You MUST double check your query before executing it. If you get an error while
            executing a query, rewrite the query and try again.

            5. You must return the coupon code, shop name, expiry date, summarize the terms and conditions.

            6. You can order the results by a relevant column to return the most interesting
            examples in the database. 

            7. Ensure all messages are user-friendly and cheerful!

            8. When a user country is specified, search for coupons for that country as well as coupons without country information.
               When a user country is not specified, you do not need to consider country in your query.            
            - 

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
            post_model_hook=self._post_model_hook,
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

    # https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/pii.py
    def _post_model_hook(self, state):
        messages = state["messages"]
        if not messages:
            return messages

        new_messages = list(messages)

        guard = Guard().use(DetectPII, ["EMAIL_ADDRESS", "PHONE_NUMBER"], "fix")

        last_ai_index = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage):
                last_ai_index = i
                break

        if last_ai_index is not None:
            for i in range(last_ai_index, len(messages)):
                msg = messages[i]
                if isinstance(msg, AIMessage):
                    ai_msg = msg
                    if not ai_msg.content:
                        continue

                    content = str(ai_msg.content)
                    print(f"Tool Message (id={ai_msg.id}): {content}")
                    result = guard.validate(content)
                    if not result.validated_output:
                        continue

                    updated_message = AIMessage(
                        content=result.validated_output,
                        id=ai_msg.id,
                        name=ai_msg.name,
                    )
                    print(f"Updated Message: {updated_message}")
                    new_messages[i] = updated_message

        return {"messages": new_messages}

    def stream(self, question, config=None, **kwargs):

        user_country = "AE" if "user_country" not in kwargs else kwargs["user_country"]
        user_prompt = "`{input}`. The user's country is {user_country}"
        query_prompt_template = ChatPromptTemplate([("human", user_prompt)])
        prompt_value = query_prompt_template.invoke(
            {"input": question, "user_country": user_country},
        )
        messages = prompt_value.to_messages()
        return self.agent_executor.stream({"messages": messages}, config, **kwargs)
