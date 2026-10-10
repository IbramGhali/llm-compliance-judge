"""Evaluation: per-rule precision/recall/F1, Cohen's kappa, evidence grounding, cost and a
regression gate. A breach is the positive class; `unclear` counts as not flagged (it goes to
a human instead), and the unclear rate is reported separately so it cannot hide errors."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from compliance_judge.models import GoldenLabel


@dataclass
class RuleScore:
    rule_id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    unclear: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class EvalResult:
    name: str
    calls: int
    pairs: int
    rules: dict[str, RuleScore]
    kappa: float
    unclear_rate: float

    @property
    def macro_precision(self) -> float:
        return statistics.fmean(s.precision for s in self.rules.values())

    @property
    def macro_recall(self) -> float:
        return statistics.fmean(s.recall for s in self.rules.values())

    @property
    def macro_f1(self) -> float:
        return statistics.fmean(s.f1 for s in self.rules.values())

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "pairs": self.pairs,
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "kappa": round(self.kappa, 4),
            "unclear_rate": round(self.unclear_rate, 4),
            "per_rule": {
                rid: {
                    "precision": round(s.precision, 3),
                    "recall": round(s.recall, 3),
                    "f1": round(s.f1, 3),
                    "tp": s.tp,
                    "fp": s.fp,
                    "fn": s.fn,
                    "tn": s.tn,
                    "unclear": s.unclear,
                }
                for rid, s in sorted(self.rules.items())
            },
        }


@dataclass
class EvalReport:
    run_id: str
    golden: EvalResult
    full: EvalResult | None
    grounding_rate: float
    quoted_breaches: int
    cost: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "golden": self.golden.summary(),
            "full": self.full.summary() if self.full else None,
            "grounding_rate": round(self.grounding_rate, 4),
            "quoted_breaches": self.quoted_breaches,
            "cost": self.cost,
        }


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float:
    """Cohen's kappa for two binary raters given (rater_a, rater_b) pairs."""
    n = len(pairs)
    if n == 0:
        return 0.0
    observed = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def score(
    name: str, labels: Mapping[tuple[str, str], str], predictions: Mapping[tuple[str, str], str]
) -> EvalResult:
    rules: dict[str, RuleScore] = {}
    pairs: list[tuple[bool, bool]] = []
    unclear = 0
    for (call_id, rule_id), label in labels.items():
        predicted = predictions.get((call_id, rule_id))
        if predicted is None:
            continue
        s = rules.setdefault(rule_id, RuleScore(rule_id))
        truth, flagged = label == "breach", predicted == "breach"
        if predicted == "unclear":
            s.unclear += 1
            unclear += 1
        if truth and flagged:
            s.tp += 1
        elif flagged:
            s.fp += 1
        elif truth:
            s.fn += 1
        else:
            s.tn += 1
        pairs.append((truth, flagged))
    predicted_calls = {cid for cid, _ in predictions}
    calls = len({cid for cid, _ in labels if cid in predicted_calls})
    return EvalResult(
        name=name,
        calls=calls,
        pairs=len(pairs),
        rules=rules,
        kappa=cohens_kappa(pairs),
        unclear_rate=unclear / len(pairs) if pairs else 0.0,
    )


def load_golden(path: Path) -> list[GoldenLabel]:
    with path.open(encoding="utf-8") as fh:
        return [GoldenLabel.model_validate_json(line) for line in fh if line.strip()]


def grounding(verdicts: list[dict[str, Any]]) -> tuple[float, int]:
    """Share of model-proposed breaches with a quote whose quote was found verbatim."""
    quoted = [v for v in verdicts if v["raw_verdict"] == "breach" and v["evidence_quote"]]
    if not quoted:
        return 1.0, 0
    found = sum(1 for v in quoted if v["post_check"] != "quote_not_found")
    return found / len(quoted), len(quoted)


def cost_summary(
    llm_calls: list[dict[str, Any]],
    pricing_path: Path,
    extraction_model: str,
    assessment_model: str,
) -> dict[str, Any]:
    with pricing_path.open(encoding="utf-8") as fh:
        prices: dict[str, dict[str, float]] = yaml.safe_load(fh)["models"]
    per_call: dict[str, dict[str, float]] = defaultdict(
        lambda: {"input": 0.0, "output": 0.0, "cost": 0.0, "latency_ms": 0.0, "calls": 0.0}
    )
    for row in llm_calls:
        model = extraction_model if row["stage"] == "extraction" else assessment_model
        price = prices.get(model, {"input": 0.0, "output": 0.0})
        c = per_call[row["call_id"]]
        c["input"] += row["input_tokens"]
        c["output"] += row["output_tokens"]
        c["cost"] += (
            row["input_tokens"] * price["input"] + row["output_tokens"] * price["output"]
        ) / 1_000_000
        c["latency_ms"] += row["latency_ms"]
        c["calls"] += 1
    if not per_call:
        return {}
    latencies = sorted(c["latency_ms"] for c in per_call.values())
    p95 = latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))]
    return {
        "models": {"extraction": extraction_model, "assessment": assessment_model},
        "llm_calls_per_transcript": statistics.fmean(c["calls"] for c in per_call.values()),
        "input_tokens_per_transcript": round(
            statistics.fmean(c["input"] for c in per_call.values())
        ),
        "output_tokens_per_transcript": round(
            statistics.fmean(c["output"] for c in per_call.values())
        ),
        "cost_usd_per_transcript": round(statistics.fmean(c["cost"] for c in per_call.values()), 5),
        "latency_ms_p50": round(statistics.median(latencies), 2),
        "latency_ms_p95": round(p95, 2),
    }


def evaluate(
    run_id: str,
    verdicts: list[dict[str, Any]],
    golden: list[GoldenLabel],
    ground_truth: dict[str, dict[str, str]] | None,
    llm_calls: list[dict[str, Any]],
    pricing_path: Path,
    extraction_model: str,
    assessment_model: str,
) -> EvalReport:
    predictions = {(v["call_id"], v["rule_id"]): v["verdict"] for v in verdicts}
    golden_labels = {(g.call_id, g.rule_id): g.label for g in golden}
    full = None
    if ground_truth:
        truth = {
            (cid, rid): lab for cid, labels in ground_truth.items() for rid, lab in labels.items()
        }
        full = score("full", truth, predictions)
    rate, quoted = grounding(verdicts)
    return EvalReport(
        run_id=run_id,
        golden=score("golden", golden_labels, predictions),
        full=full,
        grounding_rate=rate,
        quoted_breaches=quoted,
        cost=cost_summary(llm_calls, pricing_path, extraction_model, assessment_model),
    )


def check_gate(report: EvalReport, baseline_path: Path, tolerance: float) -> tuple[bool, str]:
    """Fail if golden macro F1 dropped more than `tolerance` below the committed baseline."""
    if not baseline_path.is_file():
        return False, f"no baseline at {baseline_path}; run `judge eval --update-baseline`"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_f1 = float(baseline["golden_macro_f1"])
    current = round(report.golden.macro_f1, 4)
    ok = current >= base_f1 - tolerance
    verb = "passed" if ok else "FAILED"
    return ok, (
        f"Regression gate {verb}: golden macro F1 {current:.4f} vs baseline "
        f"{base_f1:.4f} (tolerance {tolerance:.2f})"
    )


def write_baseline(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "golden_macro_f1": round(report.golden.macro_f1, 4),
        "golden_kappa": round(report.golden.kappa, 4),
        "grounding_rate": round(report.grounding_rate, 4),
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def report_json(report: EvalReport) -> str:
    return json.dumps(report.to_dict(), indent=2, default=lambda o: asdict(o))
