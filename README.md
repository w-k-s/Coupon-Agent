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

5. Setup Guardrails.ai

    1. Install [Guardrails AI](https://www.guardrailsai.com/)

        ```shell
        python3 -m pip install guardrails-ai
        ```

    2. Configure Guardrails AI with your API Key

        ```shell
        guardrails configure
        ```

    3. Important the guardrails used in this project.

        ```shell
        guardrails hub install hub://guardrails/detect_pii
        guardrails hub install hub://groundedai/grounded_ai_hallucination
        guardrails hub install hub://guardrails/profanity_free
        ```

        - `detect_pii`: There prevents leaking user data. It is possible to convince the LLM to give away data in the `users` table.

6. Seed the database using the script

    ```shell
    python seed_coupons.py
    ```

7. To run the coupon bot in RAG Agent Mode:

    ```shell
    python rag_agent.py
    ```

    To run the coupon bot in RAG Chain Mode:

    ```shell
    python rag_chain.py
    ```

## Useful Resources

- [Relevant Langchain Tutorial](https://python.langchain.com/docs/tutorials/sql_qa/)
- [Guardrails](https://www.guardrailsai.com/docs/integrations/langchain)
- [I found the example in this question useful](https://github.com/langchain-ai/langgraph/discussions/3004)

