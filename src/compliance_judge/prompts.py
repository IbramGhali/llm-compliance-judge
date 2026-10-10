"""Prompt templates are versioned files. Every verdict records the version that produced it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml

USER_MARKER = "=== USER ==="


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    version: str
    system: Template
    user: Template

    def render(self, **values: str) -> tuple[str, str]:
        return self.system.substitute(values).strip(), self.user.substitute(values).strip()


def load_prompt(path: Path) -> PromptTemplate:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path}: missing front matter")
    _, front, body = text.split("---", 2)
    meta = yaml.safe_load(front)
    if USER_MARKER not in body:
        raise ValueError(f"{path}: missing '{USER_MARKER}' separator")
    system, user = body.split(USER_MARKER, 1)
    return PromptTemplate(
        id=str(meta["id"]),
        version=str(meta["version"]),
        system=Template(system),
        user=Template(user),
    )


def load_prompts(prompts_dir: Path) -> dict[str, PromptTemplate]:
    """Load the newest version of each prompt id (file names sort by version)."""
    prompts: dict[str, PromptTemplate] = {}
    for path in sorted(prompts_dir.glob("*.md")):
        template = load_prompt(path)
        prompts[template.id] = template
    for required in ("extraction", "assessment"):
        if required not in prompts:
            raise ValueError(f"no '{required}' prompt in {prompts_dir}")
    return prompts
