"""Operational reports: per-agent scorecard and rule health (fire rate, reviewer
disagreement, drift against the previous run)."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from compliance_judge.store import Store

DRIFT_POINTS = 0.10  # alert when a rule's fire rate moves this much between runs
DRIFT_MIN_CALLS = 20


@dataclass
class RuleHealth:
    rule_id: str
    calls: int
    fire_rate: float
    unclear_rate: float
    reviewed: int
    disagreement_rate: float | None
    previous_fire_rate: float | None
    drift_alert: bool


def _rates(verdicts: list[dict[str, Any]]) -> dict[str, tuple[int, float, float]]:
    by_rule: dict[str, Counter[str]] = defaultdict(Counter)
    for v in verdicts:
        by_rule[v["rule_id"]][v["verdict"]] += 1
    out = {}
    for rule_id, counts in by_rule.items():
        n = sum(counts.values())
        out[rule_id] = (n, counts["breach"] / n, counts["unclear"] / n)
    return out


def rule_health(store: Store, run_id: str) -> list[RuleHealth]:
    current = _rates(store.verdicts(run_id))
    runs = [r["run_id"] for r in store.runs() if r["status"] == "complete"]
    previous_id = (
        next((r for r in runs[runs.index(run_id) + 1 :]), None) if run_id in runs else None
    )
    previous = _rates(store.verdicts(previous_id)) if previous_id else {}

    reviews: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in store.reviews(run_id):
        reviews[r["rule_id"]].append(r)

    health = []
    for rule_id, (n, fire, unclear) in sorted(current.items()):
        reviewed = reviews.get(rule_id, [])
        flagged = [r for r in reviewed if r["verdict"] == "breach"]
        disagreement = (
            sum(1 for r in flagged if r["label"] == "pass") / len(flagged) if flagged else None
        )
        prev = previous.get(rule_id)
        prev_fire = prev[1] if prev else None
        drift = (
            prev is not None
            and n >= DRIFT_MIN_CALLS
            and prev[0] >= DRIFT_MIN_CALLS
            and abs(fire - prev[1]) >= DRIFT_POINTS
        )
        health.append(
            RuleHealth(rule_id, n, fire, unclear, len(reviewed), disagreement, prev_fire, drift)
        )
    return health


def agent_scorecard(store: Store, run_id: str) -> list[dict[str, Any]]:
    per_agent: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"calls": set(), "breaches": 0, "unclear": 0, "rules": Counter()}
    )
    for v in store.verdicts(run_id):
        a = per_agent[v["agent"]]
        a["calls"].add(v["call_id"])
        if v["verdict"] == "breach":
            a["breaches"] += 1
            a["rules"][v["rule_id"]] += 1
        elif v["verdict"] == "unclear":
            a["unclear"] += 1
    rows = []
    for agent, a in per_agent.items():
        calls = len(a["calls"])
        rows.append(
            {
                "agent": agent,
                "calls": calls,
                "breaches": a["breaches"],
                "breaches_per_10_calls": round(10 * a["breaches"] / calls, 1),
                "unclear": a["unclear"],
                "top_rules": ", ".join(r for r, _ in a["rules"].most_common(3)),
            }
        )
    return sorted(rows, key=lambda r: r["breaches_per_10_calls"], reverse=True)
