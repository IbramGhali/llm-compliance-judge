"""Deterministic checks applied to every model output before it is stored.

The model proposes; code decides what is accepted. A breach whose evidence quote cannot be
found verbatim in the transcript is downgraded to `unclear` and sent to a human.
"""

from __future__ import annotations

from compliance_judge.models import CheckedVerdict, Fact, RuleVerdict, Transcript, Verdict
from compliance_judge.rulebook import Rule


def normalise(text: str) -> str:
    return " ".join(text.split())


def quote_in_transcript(quote: str, transcript: Transcript) -> bool:
    """True if the quote appears verbatim (whitespace-normalised) inside a single turn."""
    needle = normalise(quote)
    return bool(needle) and any(needle in normalise(t.text) for t in transcript.turns)


def ground_facts(facts: list[Fact], transcript: Transcript) -> tuple[list[Fact], int]:
    """Keep facts whose quote is in the turn they cite. Returns (kept, dropped_count)."""
    turns = {t.index: normalise(t.text) for t in transcript.turns}
    kept = [
        f for f in facts if normalise(f.quote) and normalise(f.quote) in turns.get(f.turn_index, "")
    ]
    return kept, len(facts) - len(kept)


def check_verdicts(
    raw: list[RuleVerdict], rules: list[Rule], transcript: Transcript, confidence_threshold: float
) -> list[CheckedVerdict]:
    by_rule: dict[str, RuleVerdict] = {}
    for v in raw:
        by_rule.setdefault(v.rule_id, v)  # first verdict per rule wins; extras ignored

    checked: list[CheckedVerdict] = []
    for rule in rules:
        found = by_rule.get(rule.id)
        if found is None:
            checked.append(
                CheckedVerdict(
                    rule_id=rule.id,
                    verdict=Verdict.UNCLEAR,
                    raw_verdict=Verdict.UNCLEAR,
                    rule_version=rule.version,
                    confidence=0.0,
                    rationale="Model returned no verdict for this rule.",
                    post_check="missing_verdict",
                )
            )
            continue
        v = found

        verdict, post_check = v.verdict, "ok"
        quote_ok = v.evidence_quote is None or quote_in_transcript(v.evidence_quote, transcript)
        if v.verdict is Verdict.BREACH and rule.needs_quote_for_breach and not v.evidence_quote:
            verdict, post_check = Verdict.UNCLEAR, "missing_quote"
        elif not quote_ok:
            post_check = "quote_not_found"
            if v.verdict is Verdict.BREACH:
                verdict = Verdict.UNCLEAR
        elif v.verdict is not Verdict.UNCLEAR and v.confidence < confidence_threshold:
            verdict, post_check = Verdict.UNCLEAR, "low_confidence"

        checked.append(
            CheckedVerdict(
                rule_id=rule.id,
                verdict=verdict,
                raw_verdict=v.verdict,
                rule_version=rule.version,
                evidence_quote=v.evidence_quote,
                turn_index=v.turn_index,
                confidence=v.confidence,
                rationale=v.rationale,
                post_check=post_check,
            )
        )
    return checked


def failed_verdicts(rules: list[Rule], reason: str) -> list[CheckedVerdict]:
    """Fail-safe output when a model call fails: every rule goes to human review."""
    return [
        CheckedVerdict(
            rule_id=r.id,
            verdict=Verdict.UNCLEAR,
            raw_verdict=Verdict.UNCLEAR,
            rule_version=r.version,
            confidence=0.0,
            rationale=reason,
            post_check="llm_error",
        )
        for r in rules
    ]
