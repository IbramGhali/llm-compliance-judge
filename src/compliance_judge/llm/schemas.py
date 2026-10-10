"""JSON Schemas sent to providers for structured output. Hand-written to stay within the
subset that every provider's structured-output mode accepts (no $ref, closed objects)."""

from __future__ import annotations

from typing import Any

from compliance_judge.models import FACT_LABELS


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": sorted(FACT_LABELS)},
                    "turn_index": {"type": "integer"},
                    "quote": {"type": "string"},
                    "value": _nullable({"type": "number"}),
                },
                "required": ["label", "turn_index", "quote", "value"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}


def assessment_schema(rule_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_id": {"type": "string", "enum": rule_ids},
                        "verdict": {"type": "string", "enum": ["pass", "breach", "unclear"]},
                        "evidence_quote": _nullable({"type": "string"}),
                        "turn_index": _nullable({"type": "integer"}),
                        "confidence": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "rule_id",
                        "verdict",
                        "evidence_quote",
                        "turn_index",
                        "confidence",
                        "rationale",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    }
