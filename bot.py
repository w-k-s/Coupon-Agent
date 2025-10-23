from typing import Literal, Any
from strategies.chain import CouponChain
from strategies.agent import CouponAgent
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables.config import RunnableConfig


class CouponBot:
    def __init__(
        self,
        llm,
        vector_store,
        strategy: Literal["chain", "agent"],
        coupon_db: SQLDatabase,
        allowed_tables=None,
        category_names=[],
        max_tokens=0,
        checkpointer=None,
    ):
        args = dict(
            llm=llm,
            vector_store=vector_store,
            coupon_db=coupon_db,
            allowed_tables=allowed_tables,
            category_names=category_names or [],
            max_tokens=max_tokens,
            checkpointer=checkpointer,
        )

        try:
            StrategyClass = {"chain": CouponChain, "agent": CouponAgent}[strategy]
        except KeyError:
            raise ValueError(f"Unknown strategy: {strategy}")

        self._strategy = StrategyClass(**args)

    def stream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ):
        return self._strategy.stream(input, config, **kwargs)
