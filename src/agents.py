"""Generator, Reflector, and Curator agents backed by Azure AI Agent Framework."""
from __future__ import annotations

from agent_framework import Agent
from agent_framework.azure import AzureOpenAIResponsesClient

from .models import CuratorResponse, GeneratorResponse, ReflectorResponse
from .playbook import Playbook
from .prompts import CURATOR_INSTRUCTIONS, GENERATOR_INSTRUCTIONS, REFLECTOR_INSTRUCTIONS


# ── Agent wrappers ─────────────────────────────────────────────────────────────

class GeneratorAgent:
    def __init__(self, client: AzureOpenAIResponsesClient) -> None:
        self._agent: Agent = client.as_agent(
            name="ACE-Generator",
            instructions=GENERATOR_INSTRUCTIONS,
            default_options={"response_format": GeneratorResponse},
        )

    async def generate(
        self, question: str, playbook: Playbook, reflection: str = ""
    ) -> GeneratorResponse:
        prompt = (
            f"**Playbook:**\n{playbook}\n\n"
            f"**Previous Reflection:**\n{reflection or '(none)'}\n\n"
            f"**Question:**\n{question}"
        )
        result = await self._agent.run(prompt)
        return result.value


class ReflectorAgent:
    def __init__(self, client: AzureOpenAIResponsesClient) -> None:
        self._agent: Agent = client.as_agent(
            name="ACE-Reflector",
            instructions=REFLECTOR_INSTRUCTIONS,
            default_options={"response_format": ReflectorResponse},
        )

    async def reflect(
        self, question: str, answer: str, ground_truth: str, bullets_used: str,
        prev_reflection: ReflectorResponse | None = None,
        reasoning: str = "",
    ) -> ReflectorResponse:
        feedback = (
            "Predicted answer matches ground truth."
            if not ground_truth or answer.strip() == ground_truth.strip()
            else "Predicted answer does not match ground truth."
        )
        prompt = (
            f"**Question:**\n{question}\n\n"
            f"**Model's Reasoning Trace:**\n{reasoning or '(not provided)'}\n\n"
            f"**Model's Predicted Answer:**\n{answer}\n\n"
            f"**Ground Truth Answer:**\n{ground_truth}\n\n"
            f"**Environment Feedback:**\n{feedback}\n\n"
            f"**Part of Playbook used by the generator:**\n{bullets_used}"
        )
        if prev_reflection is not None:
            prompt += (
                f"\n\n**Previous Reflection (refine this):**\n"
                f"Error: {prev_reflection.error_identification}\n"
                f"Root cause: {prev_reflection.root_cause_analysis}\n"
                f"Key insight: {prev_reflection.key_insight}"
            )
        result = await self._agent.run(prompt)
        return result.value


class CuratorAgent:
    def __init__(self, client: AzureOpenAIResponsesClient) -> None:
        self._agent: Agent = client.as_agent(
            name="ACE-Curator",
            instructions=CURATOR_INSTRUCTIONS,
            default_options={"response_format": CuratorResponse},
        )

    async def curate(
        self, playbook: Playbook, reflection: ReflectorResponse, step: int, total: int
    ) -> CuratorResponse:
        reflection_text = (
            f"Error identification: {reflection.error_identification}\n"
            f"Root cause: {reflection.root_cause_analysis}\n"
            f"Correct approach: {reflection.correct_approach}\n"
            f"Key insight: {reflection.key_insight}"
        )
        prompt = (
            f"**Training progress:** {step}/{total}\n\n"
            f"**Current Playbook:**\n{playbook}\n\n"
            f"**Reflection:**\n{reflection_text}"
        )
        result = await self._agent.run(prompt)
        return result.value
