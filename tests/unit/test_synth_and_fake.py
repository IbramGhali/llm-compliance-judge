from __future__ import annotations

from compliance_judge import synth
from compliance_judge.llm.base import LLMRequest
from compliance_judge.llm.fake import FakeLLM, extract_facts


def test_generation_is_deterministic() -> None:
    a = synth.generate(n=20, seed=5)
    b = synth.generate(n=20, seed=5)
    assert [c.transcript for c in a] == [c.transcript for c in b]
    assert [c.labels for c in a] == [c.labels for c in b]


def test_every_call_has_a_label_for_every_rule() -> None:
    for call in synth.generate(n=20, seed=5):
        assert set(call.labels) == set(synth.RULE_IDS)


def test_golden_set_covers_breaches_for_each_rule() -> None:
    calls = synth.generate(n=200, seed=7)
    golden = set(synth.select_golden(calls))
    assert len(golden) == 60
    for rule_id in synth.RULE_IDS:
        assert any(
            c.labels[rule_id] == "breach" for c in calls if c.transcript.call_id in golden
        ), rule_id


def _labels(text: str, speaker: str = "agent") -> set[str]:
    return {f["label"] for f in extract_facts([{"index": 0, "speaker": speaker, "text": text}])}


def test_fake_extraction_respects_negation() -> None:
    assert "guaranteed_outcome_claim" in _labels("I can guarantee you'll get a job.")
    assert "guaranteed_outcome_claim" not in _labels("I can't guarantee a job.")


def test_fake_extraction_reads_discount_value() -> None:
    [fact] = extract_facts(
        [{"index": 3, "speaker": "agent", "text": "I can do 20% off if you enrol with me."}]
    )
    assert fact["label"] == "discount_offered" and fact["value"] == 20.0


def test_fake_extraction_checks_speaker() -> None:
    assert "hardship_mentioned" in _labels("Money's really tight at the moment.", "customer")
    assert "hardship_mentioned" not in _labels("Money's really tight at the moment.", "agent")


def test_quote_drift_is_deterministic_and_rate_bounded() -> None:
    llm = FakeLLM(quote_drift=0.5)
    hits = [llm._drifts(f"call-{i}", "R09") for i in range(400)]
    assert hits == [FakeLLM(quote_drift=0.5)._drifts(f"call-{i}", "R09") for i in range(400)]
    assert 0.4 < sum(hits) / 400 < 0.6
    assert not any(FakeLLM(quote_drift=0.0)._drifts(f"call-{i}", "R09") for i in range(50))


def test_fake_returns_json_and_token_estimates() -> None:
    request = LLMRequest(
        task="extraction",
        model="m",
        system="s" * 40,
        user="u" * 40,
        json_schema={},
        payload={"call_id": "c", "turns": []},
    )
    response = FakeLLM().complete(request)
    assert response.text == '{"facts": []}'
    assert response.input_tokens == 20
