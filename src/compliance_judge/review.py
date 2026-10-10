"""Human review. A reviewer confirms or dismisses each flag; decisions become golden labels,
so the evaluation set grows from real disagreements instead of guesses."""

from __future__ import annotations

from pathlib import Path

from compliance_judge.evaluation import load_golden
from compliance_judge.models import GoldenLabel, LabelSource
from compliance_judge.store import Store


def decide(
    store: Store,
    verdict_id: int,
    decision: str,
    reviewer: str,
    label: str | None = None,
    comment: str | None = None,
) -> None:
    """Record a decision. For a breach, confirm means breach and dismiss means pass.
    For an unclear verdict the reviewer must give the label."""
    verdict = store.verdict(verdict_id)
    if verdict is None:
        raise KeyError(f"verdict {verdict_id} not found")
    if decision not in ("confirm", "dismiss"):
        raise ValueError("decision must be confirm or dismiss")
    if label is None:
        if verdict["verdict"] == "unclear":
            raise ValueError("unclear verdicts need an explicit label (pass or breach)")
        label = "breach" if decision == "confirm" else "pass"
    if label not in ("pass", "breach"):
        raise ValueError("label must be pass or breach")
    store.record_review(verdict_id, decision, label, reviewer, comment)


def export_to_golden(store: Store, golden_path: Path) -> tuple[int, int]:
    """Merge reviewed labels into the golden set. Human labels replace synthetic ones.
    Returns (added, replaced)."""
    existing = load_golden(golden_path) if golden_path.is_file() else []
    labels = {(g.call_id, g.rule_id): g for g in existing}
    added = replaced = 0
    for review in store.reviews():
        key = (review["call_id"], review["rule_id"])
        new = GoldenLabel(
            call_id=key[0],
            rule_id=key[1],
            label=review["label"],
            source=LabelSource.REVIEW,
            note=review["comment"],
        )
        if key in labels:
            if labels[key] != new:
                replaced += 1
        else:
            added += 1
        labels[key] = new
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with golden_path.open("w", encoding="utf-8", newline="\n") as fh:
        for key in sorted(labels):
            fh.write(labels[key].model_dump_json() + "\n")
    return added, replaced
