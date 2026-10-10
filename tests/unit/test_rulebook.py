from __future__ import annotations

import pytest
from pydantic import ValidationError

from compliance_judge.rulebook import Rule, Rulebook


def test_committed_rulebook_loads(rulebook: Rulebook) -> None:
    assert len(rulebook.rules) == 16
    assert {g.id for g in rulebook.groups} == {r.group for r in rulebook.rules}


def test_every_rule_has_a_version_and_group(rulebook: Rulebook) -> None:
    for rule in rulebook.rules:
        assert rule.version >= 1
        assert rulebook.rules_in(rule.group)


def test_quote_needed_only_for_said_things(rulebook: Rulebook) -> None:
    assert rulebook.get("R09").needs_quote_for_breach  # prohibited
    assert rulebook.get("R01").needs_quote_for_breach  # ordering
    assert not rulebook.get("R03").needs_quote_for_breach  # required: evidence of absence


def test_kind_specific_fields_are_enforced() -> None:
    with pytest.raises(ValidationError, match="needs"):
        Rule(
            id="X1",
            version=1,
            group="g",
            kind="ordering",
            severity="low",
            title="t",
            description="d",
            first="identity_confirmed",
        )


def test_unknown_fact_label_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown fact label"):
        Rule(
            id="X1",
            version=1,
            group="g",
            kind="prohibited",
            severity="low",
            title="t",
            description="d",
            forbids="made_up_label",
        )


def test_duplicate_rule_ids_are_rejected(rulebook: Rulebook) -> None:
    data = rulebook.model_dump()
    data["rules"].append(data["rules"][0])
    with pytest.raises(ValidationError, match="duplicate"):
        Rulebook.model_validate(data)
