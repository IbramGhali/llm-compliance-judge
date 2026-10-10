"""One end-to-end call against a real provider. Excluded by default; costs real money.

Run with: JUDGE_LIVE_TESTS=1 JUDGE_LLM_PROVIDER=anthropic uv run pytest -m live
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from compliance_judge import synth
from compliance_judge.app import build_app
from compliance_judge.config import Settings


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("JUDGE_LIVE_TESTS") != "1", reason="set JUDGE_LIVE_TESTS=1")
def test_one_call_end_to_end(tmp_path: Path) -> None:
    settings = replace(Settings.from_env(), db_path=tmp_path / "live.db")
    assert settings.provider != "fake", "set JUDGE_LLM_PROVIDER to a live provider"
    app = build_app(settings)
    call = synth.generate(n=1, seed=21)[0]
    app.store.ingest([call.transcript])
    summary = app.pipeline.run([call.transcript])
    assert summary.llm_errors == 0
    assert sum(summary.verdicts.values()) == 16
