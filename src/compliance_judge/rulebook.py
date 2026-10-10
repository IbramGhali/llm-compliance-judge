"""Load and validate the rulebook. Rules are versioned data, not code."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

from compliance_judge.models import FACT_LABELS

RuleKind = Literal["required", "prohibited", "ordering", "threshold", "conditional"]

# Breach verdicts for these kinds point at something that was said, so they need a quote.
# A `required` breach is evidence of absence and has no quote.
QUOTE_REQUIRED_KINDS: frozenset[str] = frozenset({"prohibited", "threshold", "ordering"})


class RuleGroup(BaseModel):
    id: str
    title: str


class Rule(BaseModel):
    id: str
    version: int
    group: str
    kind: RuleKind
    severity: Literal["low", "medium", "high"]
    title: str
    description: str
    requires: str | None = None
    applies_when: str | None = None
    within_agent_turns: int | None = None
    forbids: str | None = None
    first: str | None = None
    then: str | None = None
    fact: str | None = None
    max_value: float | None = None
    trigger: str | None = None
    unless_before: str | None = None

    @model_validator(mode="after")
    def _check_kind_fields(self) -> Rule:
        needed: dict[str, tuple[str, ...]] = {
            "required": ("requires",),
            "prohibited": ("forbids",),
            "ordering": ("first", "then"),
            "threshold": ("fact", "max_value"),
            "conditional": ("trigger", "requires", "unless_before"),
        }
        missing = [f for f in needed[self.kind] if getattr(self, f) is None]
        if missing:
            raise ValueError(f"rule {self.id}: kind '{self.kind}' needs {missing}")
        for field in (
            "requires",
            "applies_when",
            "forbids",
            "first",
            "then",
            "fact",
            "trigger",
            "unless_before",
        ):
            label = getattr(self, field)
            if label is not None and label not in FACT_LABELS:
                raise ValueError(f"rule {self.id}: unknown fact label '{label}' in {field}")
        return self

    @property
    def needs_quote_for_breach(self) -> bool:
        return self.kind in QUOTE_REQUIRED_KINDS


class Rulebook(BaseModel):
    version: str
    organisation: str
    groups: list[RuleGroup]
    rules: list[Rule]

    @model_validator(mode="after")
    def _check_ids(self) -> Rulebook:
        ids = [r.id for r in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate rule ids in rulebook")
        group_ids = {g.id for g in self.groups}
        for rule in self.rules:
            if rule.group not in group_ids:
                raise ValueError(f"rule {rule.id}: unknown group '{rule.group}'")
        return self

    def rules_in(self, group_id: str) -> list[Rule]:
        return [r for r in self.rules if r.group == group_id]

    def get(self, rule_id: str) -> Rule:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        raise KeyError(rule_id)


def load_rulebook(path: Path) -> Rulebook:
    with path.open(encoding="utf-8") as fh:
        return Rulebook.model_validate(yaml.safe_load(fh))
