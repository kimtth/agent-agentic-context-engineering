"""Self-contained finance task data processor.

Covers the formula task from the ACE finance eval:
  - formula : arithmetic finance questions, numerical answers

No dependency on the original ace/ directory.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

Task = Literal["formula"]

# ── Data paths (relative to project root) ─────────────────────────────────────

_DATA_ROOT = Path(__file__).parent / "data" / "finance"

DATA_FILES: dict[str, dict[str, Path]] = {
    "formula": {
        "train": _DATA_ROOT / "formula_train_subset_500.jsonl",
        "test":  _DATA_ROOT / "formula_test.jsonl",
    },
}


# ── JSONL loader ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(rows)} samples from {path.name}")
    return rows


# ── Context parsers ────────────────────────────────────────────────────────────

def _parse_formula(context: str) -> tuple[str, str]:
    """Extract (ctx, question) from formula-style context string."""
    if "Question: " in context and ". Answer:" in context:
        parts = context.split("Question: ", 1)
        q = parts[1].split(". Answer:")[0].strip().strip('"')
        q += (
            " Your answer should be a plain floating point number, "
            "round to the nearest hundredth if necessary. "
            "Do the necessary conversions, e.g. 5 million → 5000000.0."
        )
        return "", q
    return "", context


# ── Public API ─────────────────────────────────────────────────────────────────

def load_samples(
    task: Task,
    split: Literal["train", "test"],
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Return a list of {"question": ..., "answer": ...} dicts ready for ACE."""
    path = DATA_FILES[task][split]
    raw  = load_jsonl(path)
    if limit:
        raw = raw[:limit]

    parse = _parse_formula
    samples: list[dict[str, str]] = []

    for item in raw:
        ctx, question = parse(item.get("context", ""))
        full_question = f"{ctx}\n\n{question}".strip() if ctx else question
        samples.append({"question": full_question, "answer": item.get("target", "")})

    return samples


# ── Scoring ────────────────────────────────────────────────────────────────────

def _to_float(s: str) -> float:
    """Extract a number from a string, tolerating $, %, spaces, and surrounding text."""
    s = s.replace(",", "").replace("%", "").strip()
    # Try direct parse first
    try:
        return float(s)
    except ValueError:
        pass
    # Extract first number (including sign and decimal) from a larger string
    m = re.search(r"-?\d+\.?\d*", s)
    if m:
        return float(m.group())
    raise ValueError(f"Cannot extract number from: {s!r}")


def _formula_correct(predicted: str, ground_truth: str) -> bool:
    try:
        return _to_float(predicted) == _to_float(ground_truth)
    except ValueError:
        return predicted.strip() == ground_truth.strip()


def is_correct(task: Task):
    """Return a task-specific (predicted, ground_truth) -> bool function."""
    return _formula_correct


def evaluate_accuracy(task: Task, predictions: list[str], targets: list[str]) -> float:
    """Return overall accuracy using the task-specific scoring rule."""
    hits = sum(_formula_correct(p, t) for p, t in zip(predictions, targets))
    return hits / len(predictions) if predictions else 0.0
