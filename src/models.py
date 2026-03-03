"""Pydantic structured-output models for the three ACE agents."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class GeneratorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    answer: str
    bullet_ids_used: list[str]


class BulletTag(BaseModel):
    id: str
    tag: Literal["helpful", "harmful", "neutral"]


class ReflectorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str            # chain of thought
    error_identification: str  # what specifically went wrong
    root_cause_analysis: str   # why the error occurred
    correct_approach: str      # what should have been done instead
    key_insight: str           # principle to remember
    bullet_tags: list[BulletTag]


class PlaybookOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["ADD"]
    section: str   # e.g. "formulas_and_calculations"
    content: str   # bullet content (id will be assigned by Playbook)
    reason: str


class CuratorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    operations: list[PlaybookOperation]
