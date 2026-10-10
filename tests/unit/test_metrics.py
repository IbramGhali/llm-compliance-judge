from __future__ import annotations

import json
from pathlib import Path

import pytest

from compliance_judge.evaluation import RuleScore, check_gate, cohens_kappa, grounding, score


def test_precision_recall_f1() -> None:
    s = RuleScore("R1", tp=8, fp=2, fn=4, tn=86)
    assert s.precision == pytest.approx(0.8)
    assert s.recall == pytest.approx(8 / 12)
    assert s.f1 == pytest.approx(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12))


def test_rule_with_no_positives_and_no_flags_scores_perfectly() -> None:
    s = RuleScore("R1", tn=50)
    assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)


def test_kappa_known_value() -> None:
    # 2x2 table: both yes 20, a-only 5, b-only 10, both no 15 -> kappa = 0.4
    pairs = [(True, True)] * 20 + [(True, False)] * 5 + [(False, True)] * 10 + [(False, False)] * 15
    assert cohens_kappa(pairs) == pytest.approx(0.4)


def test_kappa_perfect_and_empty() -> None:
    assert cohens_kappa([(True, True), (False, False)]) == pytest.approx(1.0)
    assert cohens_kappa([]) == 0.0


def test_unclear_counts_as_not_flagged_but_is_reported() -> None:
    labels = {("c1", "R1"): "breach", ("c2", "R1"): "pass"}
    preds = {("c1", "R1"): "unclear", ("c2", "R1"): "pass"}
    result = score("t", labels, preds)
    assert result.rules["R1"].fn == 1 and result.rules["R1"].unclear == 1
    assert result.unclear_rate == pytest.approx(0.5)


def test_grounding_rate_counts_only_quoted_model_breaches() -> None:
    verdicts = [
        {"raw_verdict": "breach", "evidence_quote": "x", "post_check": "ok"},
        {"raw_verdict": "breach", "evidence_quote": "y", "post_check": "quote_not_found"},
        {"raw_verdict": "breach", "evidence_quote": None, "post_check": "ok"},
        {"raw_verdict": "pass", "evidence_quote": "z", "post_check": "ok"},
    ]
    assert grounding(verdicts) == (0.5, 2)


def test_gate_fails_on_regression_beyond_tolerance(tmp_path: Path) -> None:
    class R:
        class golden:  # minimal stand-in for EvalReport
            macro_f1 = 0.80

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"golden_macro_f1": 0.85}))
    ok, msg = check_gate(R, baseline, 0.02)  # type: ignore[arg-type]
    assert not ok and "FAILED" in msg
    baseline.write_text(json.dumps({"golden_macro_f1": 0.81}))
    assert check_gate(R, baseline, 0.02)[0]  # type: ignore[arg-type]
