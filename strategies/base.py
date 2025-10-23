from abc import ABC, abstractmethod
from typing import Any
from langchain_community.utilities import SQLDatabase
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages.ai import AIMessage


class BaseStrategy(ABC):

    def __init__(
        self,
        llm,
        vector_store,
        coupon_db: SQLDatabase,
        allowed_tables=None,
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

    @abstractmethod
    def stream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ):
        pass
