"""The JSON Schemas sent to providers, the Pydantic models that parse responses, and the fake
adapter's output must all agree. If one changes without the others, these tests fail."""

from __future__ import annotations

import json

import jsonschema

from compliance_judge.llm.base import LLMRequest
from compliance_judge.llm.fake import FakeLLM, extract_facts
from compliance_judge.llm.schemas import EXTRACTION_SCHEMA, assessment_schema
from compliance_judge.models import FACT_LABELS, Assessment, Extraction, Fact, RuleVerdict
from compliance_judge.rulebook import Rulebook
from compliance_judge.synth import generate


def _closed_objects(schema: object) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            found.append(schema)
        for value in schema.values():
            found.extend(_closed_objects(value))
    elif isinstance(schema, list):
        for value in schema:
            found.extend(_closed_objects(value))
    return found


def test_schemas_are_closed_and_require_every_property() -> None:
    for schema in (EXTRACTION_SCHEMA, assessment_schema(["R01"])):
        for obj in _closed_objects(schema):
            assert obj["additionalProperties"] is False
            assert set(obj["required"]) == set(obj["properties"])  # type: ignore[arg-type]


def test_schema_fields_match_models() -> None:
    fact_props = EXTRACTION_SCHEMA["properties"]["facts"]["items"]["properties"]
    assert set(fact_props) == set(Fact.model_fields)
    verdict_props = assessment_schema(["R01"])["properties"]["verdicts"]["items"]["properties"]
    assert set(verdict_props) == set(RuleVerdict.model_fields)
    assert set(fact_props["label"]["enum"]) == set(FACT_LABELS)


def test_fake_output_validates_against_schemas_and_models(rulebook: Rulebook) -> None:
    llm = FakeLLM(quote_drift=0.2)
    for call in generate(n=15, seed=9):
        turns = [t.model_dump() for t in call.transcript.turns]
        ext = llm.complete(
            LLMRequest(
                "extraction",
                "m",
                "s",
                "u",
                EXTRACTION_SCHEMA,
                {"call_id": call.transcript.call_id, "turns": turns},
            )
        )
        jsonschema.validate(json.loads(ext.text), EXTRACTION_SCHEMA)
        Extraction.model_validate_json(ext.text)
        for group in rulebook.groups:
            rules = rulebook.rules_in(group.id)
            schema = assessment_schema([r.id for r in rules])
            out = llm.complete(
                LLMRequest(
                    "assessment",
                    "m",
                    "s",
                    "u",
                    schema,
                    {
                        "call_id": call.transcript.call_id,
                        "turns": turns,
                        "facts": [
                            {k: v for k, v in f.items() if k != "_strength"}
                            for f in extract_facts(turns)
                        ],
                        "rules": [json.loads(r.model_dump_json()) for r in rules],
                    },
                )
            )
            jsonschema.validate(json.loads(out.text), schema)
            assessed = Assessment.model_validate_json(out.text)
            assert [v.rule_id for v in assessed.verdicts] == [r.id for r in rules]
