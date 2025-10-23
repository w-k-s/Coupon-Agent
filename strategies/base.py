from abc import ABC, abstractmethod
from typing import Any
from langchain_community.utilities import SQLDatabase
from langchain.globals import set_llm_cache
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_core.runnables.config import RunnableConfig
from guardrails.hub import DetectPII
from guardrails import Guard
from langchain_core.messages.ai import AIMessage


class BaseStrategy(ABC):

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

    def _create_category_retriever(self, category_names, score_threshold=0.5, k=1):
        _ = self.vector_store.add_texts(category_names)

        return self.vector_store.as_retriever(
            # search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score_threshold, "k": k},
        )

    def _apply_guard_rails(self, state):
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

    @abstractmethod
    def stream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ):
        pass
