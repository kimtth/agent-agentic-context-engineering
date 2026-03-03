"""ACE orchestrator: coordinates Generator → Reflector → Curator in a training loop."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from agent_framework.azure import AzureOpenAIResponsesClient

from .agents import CuratorAgent, GeneratorAgent, ReflectorAgent
from .playbook import Playbook


@dataclass
class StepResult:
    step: int
    question: str
    answer: str
    correct: bool
    ops_applied: int


class ACE:
    """Simplified ACE system using three Azure AI agents."""

    def __init__(
        self,
        generator: GeneratorAgent,
        reflector: ReflectorAgent,
        curator: CuratorAgent,
        playbook: Playbook | None = None,
    ) -> None:
        self.generator = generator
        self.reflector = reflector
        self.curator = curator
        self.playbook = playbook or Playbook()

    # ------------------------------------------------------------------
    @classmethod
    async def create(
        cls,
        client: AzureOpenAIResponsesClient,
        playbook: Playbook | None = None,
    ) -> "ACE":
        """Instantiate all three agents using the shared AzureOpenAIResponsesClient."""
        gen = GeneratorAgent(client)
        ref = ReflectorAgent(client)
        cur = CuratorAgent(client)
        return cls(gen, ref, cur, playbook)

    def save_playbook(self, path: str, name: str = "ACE Skill", description: str = "", title: str = "") -> None:
        """Export the trained playbook as a static SKILL.md file."""
        out = self.playbook.save_skill(path, name, description, title)
        print(f"  Playbook exported → {out}")

    # ------------------------------------------------------------------
    async def train(
        self,
        samples: list[dict[str, str]],  # each: {"question": ..., "answer": ...}
        is_correct: Callable[[str, str], bool] | None = None,
        epochs: int = 1,
        batch_size: int = 1,
        reflect_iterations: int = 1,
    ) -> list[StepResult]:
        """Run the offline training loop over labelled samples.

        Args:
            samples: list of {"question": ..., "answer": ...} dicts.
            is_correct: optional callable(predicted, ground_truth) -> bool.
                        Defaults to case-insensitive string equality.
            epochs: number of full passes over the dataset (multi-epoch adaptation).
            batch_size: samples per batch; ops are computed concurrently within a
                        batch then merged deterministically (non-LLM logic).
            reflect_iterations: Reflector refinement rounds per sample.
        """
        _is_correct = is_correct or (
            lambda pred, gt: pred.strip().lower() == gt.strip().lower()
        )
        results: list[StepResult] = []
        total = len(samples) * epochs

        for epoch in range(epochs):
            if epochs > 1:
                print(f"\n  ── Epoch {epoch + 1}/{epochs} ──")
            for batch_start in range(0, len(samples), batch_size):
                batch = samples[batch_start : batch_start + batch_size]
                step_offset = epoch * len(samples) + batch_start

                # Compute Generate→Reflect→Curate concurrently within the batch
                batch_ops = await asyncio.gather(*(
                    self._compute_ops(
                        question=s["question"],
                        ground_truth=s["answer"],
                        step=step_offset + j + 1,
                        total=total,
                        is_correct=_is_correct,
                        reflect_iterations=reflect_iterations,
                    )
                    for j, s in enumerate(batch)
                ))

                # Merge deltas deterministically (non-LLM logic)
                for j, (answer, correct, ops, bullet_tags) in enumerate(batch_ops):
                    step = step_offset + j + 1
                    self.playbook.update_bullet_counts(bullet_tags)
                    self.playbook.apply_operations(ops)
                    results.append(StepResult(
                        step=step,
                        question=batch[j]["question"],
                        answer=answer,
                        correct=correct,
                        ops_applied=len(ops),
                    ))
                    mark = "✓" if correct else "✗"
                    print(f"  [{step}/{total}] {mark}  +{len(ops)} ops")

        return results

    async def run(self, question: str) -> str:
        """Eval / inference: generate an answer with the current playbook."""
        result = await self.generator.generate(question, self.playbook)
        return result.answer

    # ------------------------------------------------------------------
    async def _compute_ops(
        self, question: str, ground_truth: str, step: int, total: int,
        is_correct: Callable[[str, str], bool],
        reflect_iterations: int = 1,
    ) -> tuple[str, bool, list, list]:
        """Generate → Reflect (×reflect_iterations) → Curate without mutating the playbook.

        Returns (answer, correct, operations, bullet_tags).
        """
        # 1. Generate trajectory
        gen = await self.generator.generate(question, self.playbook)

        # 2. Reflect – optionally refine over multiple iterations
        bullets_used = self.playbook.get_bullets_by_ids(gen.bullet_ids_used)
        ref = await self.reflector.reflect(
            question, gen.answer, ground_truth, bullets_used, reasoning=gen.reasoning,
        )
        for _ in range(reflect_iterations - 1):
            ref = await self.reflector.reflect(
                question, gen.answer, ground_truth, bullets_used,
                prev_reflection=ref, reasoning=gen.reasoning,
            )

        # 3. Curate → compact delta operations
        cur = await self.curator.curate(self.playbook, ref, step, total)

        correct = is_correct(gen.answer, ground_truth)
        return gen.answer, correct, cur.operations, ref.bullet_tags
