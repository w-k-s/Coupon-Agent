import ast
import argparse
from config import Config
from bot import CouponBot
from uuid import uuid4
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.utilities import SQLDatabase
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_community.cache import InMemoryCache


def chat(chatbot, user_country="AE"):
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

        for step in chatbot.stream(
            question, stream_mode="values", config=config, user_country=user_country
        ):
            step["messages"][-1].pretty_print()


def main():
    parser = argparse.ArgumentParser(description="Run the Coupon chatbot.")
    parser.add_argument(
        "--strategy",
        choices=["chain", "agent"],
        default="agent",
        help="Select the strategy to use: 'chain' or 'agent'. Default is 'agent'.",
    )
    args = parser.parse_args()
    print(f"Running in '{args.strategy}' mode")

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
        coupon_chatbot = CouponBot(
            llm=llm,
            cache=InMemoryCache(),
            vector_store=vector_store,
            max_tokens=384,
            checkpointer=checkpointer,
            coupon_db=coupon_db,
            allowed_tables=allowed_tables,
            category_names=categories,
            strategy=args.strategy,
        )

        chat(coupon_chatbot)


if __name__ == "__main__":
    main()
