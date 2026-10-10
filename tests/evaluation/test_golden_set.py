"""Runs the full offline pipeline over the committed golden set and applies the same
regression gate as CI: macro F1 may not drop more than the tolerance below the baseline."""

from __future__ import annotations

from dataclasses import replace

import pytest

from compliance_judge import evaluation, synth
from compliance_judge.app import build_app
from compliance_judge.config import Settings


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> evaluation.EvalReport:
    tmp = tmp_path_factory.mktemp("eval")
    settings = replace(Settings(), data_dir=tmp / "data", db_path=tmp / "judge.db")
    calls = synth.generate()  # same defaults as `judge generate`
    synth.write_dataset(calls, settings.data_dir)
    golden = evaluation.load_golden(settings.golden_path)
    golden_ids = sorted({g.call_id for g in golden})
    app = build_app(settings)
    transcripts = [c.transcript for c in calls if c.transcript.call_id in golden_ids]
    app.store.ingest(transcripts)
    run_id = app.pipeline.run(transcripts).run_id
    return evaluation.evaluate(
        run_id,
        app.store.verdicts(run_id),
        golden,
        None,
        app.store.llm_calls(run_id),
        settings.pricing_path,
        settings.extraction_model,
        settings.assessment_model,
    )


def test_golden_set_is_committed_and_complete() -> None:
    golden = evaluation.load_golden(Settings().golden_path)
    assert len({g.call_id for g in golden}) == 60
    assert len(golden) == 60 * 16


def test_regression_gate_passes(report: evaluation.EvalReport) -> None:
    ok, message = evaluation.check_gate(report, Settings().baseline_path, 0.02)
    assert ok, message


def test_every_rule_is_evaluated(report: evaluation.EvalReport) -> None:
    assert len(report.golden.rules) == 16


def test_grounding_check_catches_drifted_quotes(report: evaluation.EvalReport) -> None:
    # The fake adapter paraphrases a share of quotes; the post-check must catch them.
    assert report.quoted_breaches > 0
    assert 0.85 <= report.grounding_rate < 1.0


def test_agreement_with_labels_is_substantial(report: evaluation.EvalReport) -> None:
    assert report.golden.kappa > 0.8
