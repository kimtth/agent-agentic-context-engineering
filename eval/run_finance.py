"""
Finance evaluation harness for the simplified ACE implementation.

Usage (from project root)
-------------------------
# Quick smoke test
python -m eval.run_finance --train_size 10 --test_size 5

# Full run
python -m eval.run_finance

# Export trained playbook as a static skill
python -m eval.run_finance --train_size 50 --export-skill .github/skills/finance-formula/SKILL.md
"""
from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv

from azure.identity.aio import AzureCliCredential
from agent_framework.azure import AzureOpenAIResponsesClient

from eval.finance import evaluate_accuracy, is_correct, load_samples
from src.orchestrator import ACE

load_dotenv()


async def run(
    train_size: int | None,
    test_size: int | None,
    export_skill: str | None = None,
) -> None:
    task = "formula"
    print(f"\n{'='*60}")
    print(f"  ACE Finance Evaluation  |  task={task}")
    print(f"{'='*60}")

    train_samples = load_samples(task, "train", train_size)
    test_samples  = load_samples(task, "test",  test_size)
    correct_fn    = is_correct(task)

    print(f"  Train samples : {len(train_samples)}")
    print(f"  Test  samples : {len(test_samples)}\n")

    async with AzureCliCredential() as credential:
        client = AzureOpenAIResponsesClient(
            project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
            deployment_name=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
            credential=credential,
        )
        ace = await ACE.create(client)

        # ── Training ──────────────────────────────────────────────────────────
        print("─── Training ───────────────────────────────────────────────")
        train_results = await ace.train(train_samples, is_correct=correct_fn)
        train_hits = sum(r.correct for r in train_results)
        total_ops  = sum(r.ops_applied for r in train_results)
        print(f"\n  Train accuracy : {train_hits/len(train_results):.1%}  ({train_hits}/{len(train_results)})")
        print(f"  Playbook ops   : {total_ops}")

        # ── Final playbook snapshot ───────────────────────────────────────────
        print("\n─── Playbook ───────────────────────────────────────────────")
        print(ace.playbook)

        # ── Test evaluation ───────────────────────────────────────────────────
        print("\n─── Test Evaluation ────────────────────────────────────────")
        predictions: list[str] = []
        for i, sample in enumerate(test_samples, start=1):
            answer = await ace.run(sample["question"])
            mark   = "✓" if correct_fn(answer, sample["answer"]) else "✗"
            print(f"  [{i}/{len(test_samples)}] {mark}")
            print(f"    Q : {sample['question'][:80]}...")
            print(f"    GT: {sample['answer']}   Pred: {answer}")
            predictions.append(answer)

        targets   = [s["answer"] for s in test_samples]
        test_acc  = evaluate_accuracy(task, predictions, targets)
        test_hits = sum(correct_fn(p, t) for p, t in zip(predictions, targets))

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  SUMMARY  task={task}")
        print(f"  Train acc : {train_hits/len(train_results):.1%}")
        print(f"  Test  acc : {test_acc:.1%}  ({test_hits}/{len(targets)})")
        print(f"  Ops added : {total_ops}")
        print(f"{'='*60}\n")

        if export_skill:
            ace.save_playbook(
                export_skill,
                name="finance-formula",
                title=f"Finance Skill: {task.capitalize()} Task",
                description=(
                    f"Learned strategies for the `{task}` finance task. "
                    f"Trained on {len(train_samples)} samples."
                ),
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACE finance eval harness")
    p.add_argument("--train_size", type=int, default=None,
                   help="Max train samples (default: all)")
    p.add_argument("--test_size",  type=int, default=None,
                   help="Max test  samples (default: all)")
    p.add_argument("--export-skill", metavar="PATH", default=None,
                   help="Export trained playbook as a SKILL.md to PATH")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.train_size, args.test_size, args.export_skill))
