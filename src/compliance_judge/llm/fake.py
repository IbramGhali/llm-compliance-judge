"""Deterministic fake LLM.

It reads the structured payload a request was rendered from and behaves like a decent but
imperfect model, so offline metrics mean something:

* extraction uses a phrase lexicon that covers common wording but misses some paraphrases,
  and a short negation window that catches "I can't guarantee" but not every near miss;
* assessment applies each rule's logic to the extracted facts, with confidence taken from
  how strongly the facts matched;
* a configurable share of evidence quotes "drift" (lightly paraphrased), the failure mode
  the verbatim-quote post-check exists to catch.

Same input, same output, on every platform: no randomness, only hashlib.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any

from compliance_judge.llm.base import LLMRequest, LLMResponse, estimate_tokens


@dataclass(frozen=True)
class Pattern:
    regex: re.Pattern[str]
    strength: float
    speaker: str = "agent"


def _p(expr: str, strength: float = 0.92, speaker: str = "agent") -> Pattern:
    return Pattern(re.compile(expr, re.IGNORECASE), strength, speaker)


LEXICON: dict[str, list[Pattern]] = {
    "recording_notice": [
        _p(r"\bcalls? (is|are|is being) recorded"),
        _p(r"this call is being recorded"),
        _p(r"\brecording of this (call|conversation)", 0.8),
    ],
    "identity_confirmed": [
        _p(r"confirm your (full )?name and date of birth"),
        _p(r"verify your date of birth"),
        _p(r"date of birth", 0.74),
    ],
    "fees_discussed": [_p(r"\$\d")],
    "total_price_disclosed": [
        _p(r"total (course )?(price|cost|fee) is \$"),
        _p(r"altogether you'?d pay \$"),
        _p(r"all in, that'?s \$[\d,]+ for the whole", 0.8),
    ],
    "payment_plan_offered": [_p(r"payment plan"), _p(r"monthly payments", 0.8)],
    "instalment_terms_stated": [
        _p(r"\d+ (monthly )?instalments of \$"),
        _p(r"\$[\d,]+ a month over \d+ months"),
        _p(r"split into \w+ payments of \$", 0.8),
    ],
    "discount_offered": [_p(r"(\d+)% off")],
    "entry_requirements_stated": [
        _p(r"entry requirement"),
        _p(r"(finished|completed) year 12", 0.8),
        _p(r"to get in, you need", 0.74),
    ],
    "duration_stated": [
        _p(r"\d+ hours a week"),
        _p(r"finish in \d+ months"),
        _p(r"takes around a year", 0.74),
        _p(r"hours of study each week", 0.8),
    ],
    "cooling_off_disclosed": [
        _p(r"(10|ten)[- ]day cooling[- ]off"),
        _p(r"within 10 days.*cancel", 0.8),
    ],
    "refund_policy_explained": [
        _p(r"refunds? (are|is) pro-?rata"),
        _p(r"we refund the units", 0.8),
    ],
    "guaranteed_outcome_claim": [_p(r"guarantee"), _p(r"definitely get you hired")],
    "false_urgency_claim": [
        _p(r"only available if you sign up today"),
        _p(r"ends tonight"),
        _p(r"decide right now"),
    ],
    "accreditation_claim": [_p(r"university degree"), _p(r"bachelor'?s degree")],
    "salary_claim": [_p(r"earn \$[\d,]+"), _p(r"six figures")],
    "cancellation_discouraged": [_p(r"can'?t (really )?cancel"), _p(r"cancelling is a hassle")],
    "hardship_mentioned": [
        _p(r"money'?s (really )?tight", speaker="customer"),
        _p(r"can afford", 0.8, speaker="customer"),
    ],
    "callback_offered": [_p(r"call you back")],
    "enrolment_consent_given": [
        _p(r"\bI agree\b", speaker="customer"),
        _p(r"go ahead", 0.85, speaker="customer"),
        _p(r"sign me up", speaker="customer"),
    ],
    "enrolment_processed": [
        _p(r"processed your enrolment"),
        _p(r"(all done|now), you'?re enrolled"),
    ],
}
# Claims where a nearby negation ("can't", "not") means the claim was not made.
NEGATABLE = {
    "guaranteed_outcome_claim",
    "false_urgency_claim",
    "accreditation_claim",
    "salary_claim",
}
NEGATIONS = {"not", "no", "never", "can't", "cannot", "won't", "don't", "isn't"}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _negated(sentence: str, start: int) -> bool:
    before = re.findall(r"[\w']+", sentence[:start].lower())[-3:]
    return any(word in NEGATIONS for word in before)


def extract_facts(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for turn in turns:
        for sentence in SENTENCE_SPLIT.split(turn["text"]):
            for label, patterns in LEXICON.items():
                for pattern in patterns:
                    if pattern.speaker != turn["speaker"]:
                        continue
                    match = pattern.regex.search(sentence)
                    if not match or (label in NEGATABLE and _negated(sentence, match.start())):
                        continue
                    value = float(match.group(1)) if label == "discount_offered" else None
                    facts.append(
                        {
                            "label": label,
                            "turn_index": turn["index"],
                            "quote": sentence,
                            "value": value,
                            "_strength": pattern.strength,
                        }
                    )
                    break
    return facts


def _strength_of(fact: dict[str, Any]) -> float:
    """Confidence for a fact: the strength of the first lexicon pattern its quote matches."""
    for pattern in LEXICON.get(fact["label"], []):
        if pattern.regex.search(fact["quote"]):
            return pattern.strength
    return 0.7


def _first(facts: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    hits = [f for f in facts if f["label"] == label]
    return min(hits, key=lambda f: f["turn_index"]) if hits else None


def _verdict(
    rule_id: str,
    verdict: str,
    confidence: float,
    rationale: str,
    fact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "verdict": verdict,
        "evidence_quote": fact["quote"] if fact else None,
        "turn_index": fact["turn_index"] if fact else None,
        "confidence": round(confidence, 2),
        "rationale": rationale,
    }


def assess_rule(
    rule: dict[str, Any], facts: list[dict[str, Any]], turns: list[dict[str, Any]]
) -> dict[str, Any]:
    rid, kind = rule["id"], rule["kind"]
    if kind == "prohibited":
        hit = _first(facts, rule["forbids"])
        if hit:
            return _verdict(
                rid, "breach", hit["_strength"], f"Agent made a {rule['forbids']}.", hit
            )
        return _verdict(rid, "pass", 0.85, "No prohibited claim found.")
    if kind == "required":
        if rule.get("applies_when") and not _first(facts, rule["applies_when"]):
            return _verdict(rid, "pass", 0.9, f"Not applicable: no {rule['applies_when']}.")
        hit = _first(facts, rule["requires"])
        limit = rule.get("within_agent_turns")
        if hit and limit:
            agent_turns = [t["index"] for t in turns if t["speaker"] == "agent"][:limit]
            if hit["turn_index"] not in agent_turns:
                return _verdict(
                    rid,
                    "breach",
                    0.8,
                    f"{rule['requires']} came after the first {limit} agent turns.",
                    hit,
                )
        if hit:
            return _verdict(rid, "pass", hit["_strength"], f"Found {rule['requires']}.", hit)
        return _verdict(rid, "breach", 0.8, f"No {rule['requires']} found in the call.")
    if kind == "ordering":
        then = _first(facts, rule["then"])
        if not then:
            return _verdict(rid, "pass", 0.9, f"Not applicable: no {rule['then']}.")
        first = _first(facts, rule["first"])
        if first and first["turn_index"] < then["turn_index"]:
            return _verdict(
                rid,
                "pass",
                min(first["_strength"], then["_strength"]),
                f"{rule['first']} came before {rule['then']}.",
            )
        return _verdict(
            rid,
            "breach",
            then["_strength"] * 0.9,
            f"{rule['then']} happened before {rule['first']}.",
            then,
        )
    if kind == "threshold":
        offers = [f for f in facts if f["label"] == rule["fact"] and f["value"] is not None]
        over = [f for f in offers if f["value"] > rule["max_value"]]
        if over:
            return _verdict(
                rid,
                "breach",
                over[0]["_strength"],
                f"Offered {over[0]['value']:g}% (limit {rule['max_value']:g}%).",
                over[0],
            )
        return _verdict(rid, "pass", 0.85, "No offer above the limit.")
    # conditional
    trigger = _first(facts, rule["trigger"])
    if not trigger:
        return _verdict(rid, "pass", 0.85, f"Not applicable: no {rule['trigger']}.")
    stop = _first(facts, rule["unless_before"])
    response = [
        f
        for f in facts
        if f["label"] == rule["requires"]
        and f["turn_index"] > trigger["turn_index"]
        and (stop is None or f["turn_index"] < stop["turn_index"])
    ]
    if response or stop is None:
        return _verdict(rid, "pass", trigger["_strength"], "Hardship was responded to.")
    return _verdict(
        rid,
        "breach",
        trigger["_strength"] * 0.9,
        f"{rule['unless_before']} without {rule['requires']} after hardship.",
        stop,
    )


def _drift(quote: str) -> str:
    words = quote.split()
    if len(words) > 3:
        del words[len(words) // 2]
    return " ".join(words).lower()


class FakeLLM:
    """Offline adapter. `quote_drift` is the share of evidence quotes it paraphrases."""

    name = "fake"

    def __init__(self, quote_drift: float = 0.03) -> None:
        self.quote_drift = quote_drift

    def _drifts(self, *keys: str) -> bool:
        digest = hashlib.sha256("|".join(keys).encode()).digest()
        return int.from_bytes(digest[:4], "big") / 2**32 < self.quote_drift

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        payload = request.payload
        if request.task == "extraction":
            facts = extract_facts(payload["turns"])
            body: dict[str, Any] = {
                "facts": [{k: v for k, v in f.items() if k != "_strength"} for f in facts]
            }
        else:
            # Assess the facts stage 1 produced, as a real model would.
            facts = [{**f, "_strength": _strength_of(f)} for f in payload["facts"]]
            verdicts = []
            for rule in payload["rules"]:
                v = assess_rule(rule, facts, payload["turns"])
                if v["evidence_quote"] and self._drifts(payload["call_id"], rule["id"]):
                    v["evidence_quote"] = _drift(v["evidence_quote"])
                verdicts.append(v)
            body = {"verdicts": verdicts}
        text = json.dumps(body)
        return LLMResponse(
            text=text,
            model=f"fake:{request.model}",
            input_tokens=estimate_tokens(request.system + request.user),
            output_tokens=estimate_tokens(text),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
