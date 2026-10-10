from __future__ import annotations

import pytest

from compliance_judge import review, synth
from compliance_judge.app import App
from compliance_judge.evaluation import load_golden
from compliance_judge.models import LabelSource


def test_ingest_is_idempotent(small_app: App) -> None:
    transcripts = synth.load_transcripts(small_app.settings.data_dir)
    assert small_app.store.ingest(transcripts) == (30, 0)
    assert small_app.store.ingest(transcripts) == (0, 30)
    changed = transcripts[0].model_copy(update={"agent": "Someone Else"})
    assert small_app.store.ingest([changed]) == (1, 0)
    assert len(small_app.store.transcripts()) == 30


def test_rerunning_a_run_does_not_duplicate_verdicts(small_app: App) -> None:
    transcripts = synth.load_transcripts(small_app.settings.data_dir)[:5]
    small_app.store.ingest(transcripts)
    run_id = small_app.pipeline.start_run()
    for t in transcripts:
        facts, verdicts, _, _ = small_app.pipeline.process_call(run_id, t)
        small_app.store.save_call_result(run_id, t.call_id, facts, verdicts)
        small_app.store.save_call_result(run_id, t.call_id, facts, verdicts)
    assert len(small_app.store.verdicts(run_id)) == 5 * 16


def test_resume_skips_completed_calls(small_app: App) -> None:
    transcripts = synth.load_transcripts(small_app.settings.data_dir)[:6]
    small_app.store.ingest(transcripts)
    first = small_app.pipeline.run(transcripts[:4])
    resumed = small_app.pipeline.run(transcripts, run_id=first.run_id)
    assert (resumed.skipped, resumed.calls) == (4, 2)


def test_every_verdict_records_its_provenance(small_app: App) -> None:
    transcripts = synth.load_transcripts(small_app.settings.data_dir)[:2]
    small_app.store.ingest(transcripts)
    run_id = small_app.pipeline.run(transcripts).run_id
    run = small_app.store.run(run_id)
    assert run is not None
    assert run["extraction_prompt"] == "extraction.v1"
    assert run["rulebook_version"] == small_app.rulebook.version
    assert all(v["rule_version"] >= 1 for v in small_app.store.verdicts(run_id))


def test_review_decisions_become_golden_labels(small_app: App) -> None:
    transcripts = synth.load_transcripts(small_app.settings.data_dir)
    small_app.store.ingest(transcripts)
    run_id = small_app.pipeline.run(transcripts).run_id
    queue = small_app.store.review_queue(run_id)
    assert queue, "expected flagged verdicts in the review queue"
    target = next(v for v in queue if v["verdict"] == "breach")
    review.decide(small_app.store, target["id"], "dismiss", "tester", comment="near miss")

    added, replaced = review.export_to_golden(small_app.store, small_app.settings.golden_path)
    labels = {(g.call_id, g.rule_id): g for g in load_golden(small_app.settings.golden_path)}
    label = labels[(target["call_id"], target["rule_id"])]
    assert label.label == "pass" and label.source is LabelSource.REVIEW
    assert added + replaced == 1
    assert all(v["id"] != target["id"] for v in small_app.store.review_queue(run_id))


def test_unclear_review_requires_label(small_app: App) -> None:
    transcripts = synth.load_transcripts(small_app.settings.data_dir)[:3]
    small_app.store.ingest(transcripts)
    run_id = small_app.pipeline.run(transcripts).run_id
    verdict = small_app.store.verdicts(run_id)[0]
    small_app.store.conn.execute(
        "UPDATE verdicts SET verdict='unclear' WHERE id=?", (verdict["id"],)
    )
    with pytest.raises(ValueError, match="explicit label"):
        review.decide(small_app.store, verdict["id"], "confirm", "tester")
    review.decide(small_app.store, verdict["id"], "confirm", "tester", label="breach")
