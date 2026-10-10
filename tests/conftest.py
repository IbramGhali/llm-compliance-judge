from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from compliance_judge import synth
from compliance_judge.app import App, build_app
from compliance_judge.config import Settings
from compliance_judge.models import Transcript, Turn
from compliance_judge.rulebook import Rulebook, load_rulebook


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Offline settings with every writable path inside tmp_path."""
    return replace(
        Settings(),
        data_dir=tmp_path / "data",
        db_path=tmp_path / "judge.db",
        golden_path=tmp_path / "golden.jsonl",
        baseline_path=tmp_path / "eval" / "baseline.json",
    )


@pytest.fixture
def rulebook() -> Rulebook:
    return load_rulebook(Settings().rulebook_path)


@pytest.fixture
def small_app(settings: Settings) -> App:
    calls = synth.generate(n=30, seed=3)
    synth.write_dataset(calls, settings.data_dir)
    synth.write_golden(calls, synth.select_golden(calls, size=15), settings.golden_path)
    return build_app(settings)


def make_transcript(*lines: tuple[str, str], call_id: str = "call-test") -> Transcript:
    turns = [
        Turn(index=i, speaker=s, t=float(i), text=text)  # type: ignore[arg-type]
        for i, (s, text) in enumerate(lines)
    ]
    return Transcript(
        call_id=call_id,
        agent="Alex",
        customer="Test Person",
        course="Test Course",
        started_at="2026-07-01T09:00:00",
        turns=turns,
    )
