"""
Simplified ACE (Agentic Context Engineering) — entry point.

Prerequisites
-------------
1. Install dependencies:
       pip install -e .

2. Set environment variables:
       AZURE_AI_PROJECT_ENDPOINT   https://<project>.services.ai.azure.com/api/projects/<id>
       AZURE_AI_MODEL_DEPLOYMENT_NAME  gpt-4o-mini  (or your deployment)

3. Authenticate:
       az login
"""
import asyncio
import os

from dotenv import load_dotenv

from azure.identity.aio import AzureCliCredential
from agent_framework.azure import AzureOpenAIResponsesClient

from src.orchestrator import ACE

load_dotenv()

# ── Sample data ────────────────────────────────────────────────────────────────

TRAIN_SAMPLES = [
    {
        "question": "What is compound interest and its formula?",
        "answer": "Compound interest earns interest on previous interest. Formula: A = P(1 + r/n)^(nt)",
    },
    {
        "question": "How is Net Present Value (NPV) calculated?",
        "answer": "NPV = Σ [CF_t / (1+r)^t] - Initial Investment",
    },
]

TEST_QUESTIONS = [
    "$1,000 invested at 5% annual interest compounded monthly for 3 years – what is the final amount?",
]


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    async with AzureCliCredential() as credential:
        client = AzureOpenAIResponsesClient(
            project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
            deployment_name=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
            credential=credential,
        )
        ace = await ACE.create(client)

        print("=== Training ===")
        results = await ace.train(TRAIN_SAMPLES)
        correct = sum(r.correct for r in results)
        print(f"Accuracy: {correct}/{len(results)}\n")

        print("--- Playbook ---")
        print(ace.playbook)

        print("\n=== Evaluation ===")
        for question in TEST_QUESTIONS:
            answer = await ace.run(question)
            print(f"Q: {question}\nA: {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())

