from __future__ import annotations

from compliance_judge.models import Fact, RuleVerdict, Verdict
from compliance_judge.pipeline.postchecks import (
    check_verdicts,
    failed_verdicts,
    ground_facts,
    quote_in_transcript,
)
from compliance_judge.rulebook import Rulebook
from tests.conftest import make_transcript

T = make_transcript(
    ("agent", "Hi there.  I can guarantee you'll get a job when you finish."),
    ("customer", "Really?"),
)


def _v(rule_id: str, verdict: str, quote: str | None, confidence: float = 0.9) -> RuleVerdict:
    return RuleVerdict(
        rule_id=rule_id,
        verdict=Verdict(verdict),
        evidence_quote=quote,
        turn_index=0 if quote else None,
        confidence=confidence,
        rationale="r",
    )


def test_quote_match_is_whitespace_normalised_but_otherwise_exact() -> None:
    assert quote_in_transcript("Hi there. I can guarantee", T)
    assert not quote_in_transcript("i can guarantee you will get a job", T)
    assert not quote_in_transcript("", T)


def test_breach_with_verbatim_quote_is_kept(rulebook: Rulebook) -> None:
    [v] = check_verdicts(
        [_v("R09", "breach", "I can guarantee you'll get a job when you finish.")],
        [rulebook.get("R09")],
        T,
        0.6,
    )
    assert v.verdict is Verdict.BREACH and v.post_check == "ok"


def test_breach_with_paraphrased_quote_is_downgraded(rulebook: Rulebook) -> None:
    [v] = check_verdicts(
        [_v("R09", "breach", "I guarantee you will get a job.")], [rulebook.get("R09")], T, 0.6
    )
    assert v.verdict is Verdict.UNCLEAR
    assert v.raw_verdict is Verdict.BREACH
    assert v.post_check == "quote_not_found"


def test_breach_without_quote_is_downgraded_when_quote_required(rulebook: Rulebook) -> None:
    [v] = check_verdicts([_v("R09", "breach", None)], [rulebook.get("R09")], T, 0.6)
    assert v.verdict is Verdict.UNCLEAR and v.post_check == "missing_quote"


def test_required_disclosure_breach_needs_no_quote(rulebook: Rulebook) -> None:
    [v] = check_verdicts([_v("R03", "breach", None)], [rulebook.get("R03")], T, 0.6)
    assert v.verdict is Verdict.BREACH


def test_low_confidence_goes_to_review(rulebook: Rulebook) -> None:
    [v] = check_verdicts([_v("R03", "pass", None, confidence=0.4)], [rulebook.get("R03")], T, 0.6)
    assert v.verdict is Verdict.UNCLEAR and v.post_check == "low_confidence"


def test_missing_verdict_becomes_unclear_and_extras_are_ignored(rulebook: Rulebook) -> None:
    rules = [rulebook.get("R09"), rulebook.get("R10")]
    out = check_verdicts(
        [_v("R09", "pass", None), _v("R09", "breach", None), _v("R99", "breach", None)],
        rules,
        T,
        0.6,
    )
    assert [(v.rule_id, v.verdict) for v in out] == [
        ("R09", Verdict.PASS),
        ("R10", Verdict.UNCLEAR),
    ]
    assert out[1].post_check == "missing_verdict"


def test_verdict_records_rule_version(rulebook: Rulebook) -> None:
    [v] = check_verdicts([_v("R04", "pass", None)], [rulebook.get("R04")], T, 0.6)
    assert v.rule_version == 2


def test_failed_call_sends_every_rule_to_review(rulebook: Rulebook) -> None:
    out = failed_verdicts(rulebook.rules, "timeout")
    assert len(out) == 16 and all(v.verdict is Verdict.UNCLEAR for v in out)


def test_facts_must_quote_the_turn_they_cite() -> None:
    facts = [
        Fact(label="guaranteed_outcome_claim", turn_index=0, quote="I can guarantee"),
        Fact(label="guaranteed_outcome_claim", turn_index=1, quote="I can guarantee"),
    ]
    kept, dropped = ground_facts(facts, T)
    assert len(kept) == 1 and dropped == 1
