# Coupon Agent

**TODO**
Either: 
- Refactor `seed_coupons.py`
- just commit as large as possible db.
- Use github lfs

## Goals

| Feature                                   | RAG Chatbot | RAG Agent |
|-------------------------------------------|-------------|-----------|
| 1. Find Coupons by shop name              | ✅          | ✅        |
| 2. Find Coupons by category               | ✅          | ✅        |
| 3. Find Coupons that are not expired      | ✅          | ✅        |
| 4. Find Coupons for user's country        | ✅          | ✅        |
| 5. Use Memory                             | ✅          | ✅        |
| 6. Trim memory                            | ✅          | ✅        |
| 7. Caching                                | ✅          | ✅        |
| 9. Code Structure                         | ✅          | ✅        |
| 10. Setup Guard rails                     | ❌          | ❌        |
| 11. Error Handling                        | ❌          | ❌        |
| 12. Deployment                            | ❌          | ❌        |
| 13. Option (Trim vs. Summarize)           | ❌          | ❌        |

## Setup

**Prerequisites**
- Python 3.12

1. Create a virtual env (if it doesn't already exist)

    ```shell
    python3 -m venv .venv
    ```

2. Activate the virtual env

    ```shell
    source .venv/bin/activate
    ```

3. Sync dependencies

    ```shell
    python3 -m pip install -r requirements.txt
    ```

4. Create an `.env` file based on the `.env.example`

    ```shell
    cp .env.example .env
    ```

    Update with your `OPENAI_API_KEY`.

5. Seed the database using the script

    ```shell
    python seed_coupons.py
    ```

6. To run the coupon bot in RAG Agent Mode:

    ```shell
    python rag_agent.py
    ```

    To run the coupon bot in RAG Chain Mode:

    ```shell
    python rag_chain.py
    ```

## Useful Resources

![I found the example in this question useful](https://github.com/langchain-ai/langgraph/discussions/3004)

