from __future__ import annotations

from pathlib import Path

import pytest

from compliance_judge.config import Settings
from compliance_judge.prompts import load_prompt, load_prompts


def test_committed_prompts_load_with_versions() -> None:
    prompts = load_prompts(Settings().prompts_dir)
    assert prompts["extraction"].version == "extraction.v1"
    assert prompts["assessment"].version == "assessment.v1"


def test_render_fills_every_placeholder() -> None:
    prompts = load_prompts(Settings().prompts_dir)
    system, user = prompts["extraction"].render(
        labels="- a: b", call_id="c1", course="X", transcript="[0] agent: hi"
    )
    assert "$" not in system and "[0] agent: hi" in user


def test_missing_placeholder_value_raises() -> None:
    prompts = load_prompts(Settings().prompts_dir)
    with pytest.raises(KeyError):
        prompts["extraction"].render(call_id="c1")


def test_prompt_without_separator_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.md"
    bad.write_text("---\nid: x\nversion: x.v1\n---\nno separator here\n")
    with pytest.raises(ValueError, match="separator"):
        load_prompt(bad)
