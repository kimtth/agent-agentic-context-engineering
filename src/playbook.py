"""Playbook: a versioned, section-based knowledge store for ACE."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from .models import PlaybookOperation

# section key → (slug, display header)
_SECTIONS: dict[str, tuple[str, str]] = {
    "strategies_and_insights":    ("str",  "STRATEGIES & INSIGHTS"),
    "formulas_and_calculations":  ("calc", "FORMULAS & CALCULATIONS"),
    "common_mistakes_to_avoid":   ("err",  "COMMON MISTAKES TO AVOID"),
    "problem_solving_heuristics": ("prob", "PROBLEM-SOLVING HEURISTICS"),
}

_BULLET_RE = re.compile(r"\[([^\]]+)\]\s*helpful=\d+\s*harmful=\d+\s*::\s*(.*)")
_COUNTS_RE = re.compile(r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::(.*)")


def _parse_counts(line: str) -> tuple[str, int, int, str] | None:
    m = _COUNTS_RE.match(line.strip())
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)), m.group(4).strip()
    return None


class Playbook:
    def __init__(self, content: str = "") -> None:
        self._content = content or self._empty()
        self._next_id = 1

    # ------------------------------------------------------------------
    def apply_operations(self, operations: list[PlaybookOperation]) -> None:
        for op in operations:
            if op.type == "ADD":
                self._add(op.section, op.content)

    def update_bullet_counts(self, bullet_tags: list) -> None:
        """Increment helpful/harmful counters based on Reflector tags."""
        tag_map = {bt.id: bt.tag for bt in bullet_tags}
        if not tag_map:
            return
        lines = self._content.splitlines()
        updated: list[str] = []
        for line in lines:
            m = _BULLET_RE.match(line.strip())
            if m and m.group(1) in tag_map:
                parsed = _parse_counts(line)
                if parsed:
                    bid, helpful, harmful, content = parsed
                    tag = tag_map[bid]
                    if tag == "helpful":
                        helpful += 1
                    elif tag == "harmful":
                        harmful += 1
                    updated.append(f"[{bid}] helpful={helpful} harmful={harmful} :: {content}")
                else:
                    updated.append(line)
            else:
                updated.append(line)
        self._content = "\n".join(updated)

    def get_bullets_by_ids(self, ids: list[str]) -> str:
        if not ids:
            return "(none)"
        found = []
        for line in self._content.splitlines():
            m = _BULLET_RE.match(line.strip())
            if m and m.group(1) in ids:
                found.append(line.strip())
        return "\n".join(found) or "(none)"

    def __str__(self) -> str:
        return self._content

    # ------------------------------------------------------------------
    def _add(self, section_key: str, content: str) -> None:
        slug, header = _SECTIONS.get(section_key, (section_key[:4], section_key.upper()))
        bullet_id = f"{slug}-{self._next_id:05d}"
        self._next_id += 1
        new_line = f"[{bullet_id}] helpful=0 harmful=0 :: {content}"
        marker = f"## {header}"
        if marker in self._content:
            self._content = self._content.replace(marker, f"{marker}\n{new_line}", 1)
        else:
            self._content += f"\n{new_line}"

    # ------------------------------------------------------------------
    def export_skill(self, name: str, description: str = "", title: str = "") -> str:
        """Render the playbook as a SKILL.md-compatible static config string."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        bullets = [
            line.strip()
            for line in self._content.splitlines()
            if _BULLET_RE.match(line.strip())
        ]
        # Strip ACE metadata (helpful=/harmful= counters) for clean export
        clean: list[str] = []
        for b in bullets:
            m = _BULLET_RE.match(b)
            if m:
                clean.append(f"- {m.group(2).strip()}")

        sections: dict[str, list[str]] = {header: [] for _, header in _SECTIONS.values()}
        for line in self._content.splitlines():
            current: str | None = None
            stripped = line.strip()
            if stripped.startswith("## "):
                current_header = stripped[3:]
                if current_header in sections:
                    current = current_header
            elif current and _BULLET_RE.match(stripped):
                m = _BULLET_RE.match(stripped)
                if m:
                    sections[current].append(f"- {m.group(2).strip()}")

        # Rebuild sections in order
        section_blocks: list[str] = []
        current_header: str | None = None
        current_items: list[str] = []
        for line in self._content.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                if current_header and current_items:
                    section_blocks.append(f"### {current_header}\n" + "\n".join(current_items))
                current_header = stripped[3:]
                current_items = []
            elif current_header:
                m = _BULLET_RE.match(stripped)
                if m:
                    current_items.append(f"- {m.group(2).strip()}")
        if current_header and current_items:
            section_blocks.append(f"### {current_header}\n" + "\n".join(current_items))

        body = "\n\n".join(section_blocks) if section_blocks else "*(no bullets learned yet)*"
        desc_line = f"\n{description}\n" if description else ""
        frontmatter = f"---\nname: {name}\n"
        if description:
            frontmatter += f"description: {description}\n"
        frontmatter += "---\n\n"
        heading = title or name
        return (
            f"{frontmatter}"
            f"# {heading}\n"
            f"{desc_line}\n"
            f"*Generated by ACE on {ts}. "
            f"Total bullets: {len(clean)}.*\n\n"
            f"{body}\n"
        )

    def save_skill(self, path: str | Path, name: str, description: str = "", title: str = "") -> Path:
        """Write export_skill() output to *path* and return the resolved Path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.export_skill(name, description, title), encoding="utf-8")
        return out

    @staticmethod
    def _empty() -> str:
        return "\n\n".join(
            f"## {header}"
            for _, header in _SECTIONS.values()
        )
