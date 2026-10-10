"""Two-stage pipeline: extract what was said, then assess each rule group against it."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field

from pydantic import ValidationError

from compliance_judge.config import Settings
from compliance_judge.llm import LLMClient, LLMError, LLMRequest, LLMResponse
from compliance_judge.llm.schemas import EXTRACTION_SCHEMA, assessment_schema
from compliance_judge.models import (
    FACT_LABELS,
    Assessment,
    CheckedVerdict,
    Extraction,
    Fact,
    Transcript,
)
from compliance_judge.pipeline.postchecks import check_verdicts, failed_verdicts, ground_facts
from compliance_judge.prompts import PromptTemplate
from compliance_judge.rulebook import Rule, Rulebook
from compliance_judge.store import Store


@dataclass
class RunSummary:
    run_id: str
    calls: int = 0
    skipped: int = 0
    verdicts: Counter[str] = field(default_factory=Counter)
    post_checks: Counter[str] = field(default_factory=Counter)
    facts_dropped: int = 0
    llm_errors: int = 0
    seconds: float = 0.0


def format_transcript(t: Transcript) -> str:
    return "\n".join(f"[{turn.index}] {turn.speaker}: {turn.text}" for turn in t.turns)


def _rule_line(rule: Rule) -> str:
    return f"- {rule.id} ({rule.kind}, {rule.severity} severity): {rule.title}. {rule.description}"


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        rulebook: Rulebook,
        prompts: dict[str, PromptTemplate],
        store: Store,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.rulebook = rulebook
        self.extraction_prompt = prompts["extraction"]
        self.assessment_prompt = prompts["assessment"]
        self.store = store

    def start_run(self) -> str:
        return self.store.create_run(
            adapter=self.llm.name,
            extraction_model=self.settings.extraction_model,
            assessment_model=self.settings.assessment_model,
            extraction_prompt=self.extraction_prompt.version,
            assessment_prompt=self.assessment_prompt.version,
            rulebook_version=self.rulebook.version,
        )

    def run(self, transcripts: list[Transcript], run_id: str | None = None) -> RunSummary:
        started = time.perf_counter()
        run_id = run_id or self.start_run()
        summary = RunSummary(run_id=run_id)
        done = self.store.done_calls(run_id)
        for transcript in transcripts:
            if transcript.call_id in done:
                summary.skipped += 1
                continue
            facts, verdicts, dropped, errors = self.process_call(run_id, transcript)
            self.store.save_call_result(run_id, transcript.call_id, facts, verdicts)
            summary.calls += 1
            summary.facts_dropped += dropped
            summary.llm_errors += errors
            summary.verdicts.update(v.verdict.value for v in verdicts)
            summary.post_checks.update(v.post_check for v in verdicts if v.post_check != "ok")
        self.store.finish_run(run_id)
        summary.seconds = time.perf_counter() - started
        return summary

    def _call(
        self, run_id: str, call_id: str, stage: str, request: LLMRequest
    ) -> LLMResponse | None:
        try:
            response = self.llm.complete(request)
        except LLMError:
            self.store.log_llm_call(run_id, call_id, stage, request.model, 0, 0, 0.0, False)
            return None
        self.store.log_llm_call(
            run_id,
            call_id,
            stage,
            request.model,
            response.input_tokens,
            response.output_tokens,
            response.latency_ms,
            True,
        )
        return response

    def process_call(
        self, run_id: str, transcript: Transcript
    ) -> tuple[list[Fact], list[CheckedVerdict], int, int]:
        turns = [t.model_dump() for t in transcript.turns]
        text = format_transcript(transcript)
        errors = 0

        # Stage 1: extraction (cheap model). What was said, with quotes.
        system, user = self.extraction_prompt.render(
            labels="\n".join(f"- {k}: {v}" for k, v in FACT_LABELS.items()),
            call_id=transcript.call_id,
            course=transcript.course,
            transcript=text,
        )
        response = self._call(
            run_id,
            transcript.call_id,
            "extraction",
            LLMRequest(
                task="extraction",
                model=self.settings.extraction_model,
                system=system,
                user=user,
                json_schema=EXTRACTION_SCHEMA,
                payload={"call_id": transcript.call_id, "turns": turns},
            ),
        )
        try:
            if response is None:
                raise LLMError("extraction call failed")
            facts = Extraction.model_validate_json(response.text).facts
        except (LLMError, ValidationError):
            errors += 1
            return [], failed_verdicts(self.rulebook.rules, "Extraction failed."), 0, errors
        facts, dropped = ground_facts(facts, transcript)

        # Stage 2: assessment per rule group (stronger model). Was a rule broken?
        facts_text = (
            "\n".join(
                f'- {f.label}, turn {f.turn_index}, "{f.quote}"'
                + (f", value {f.value:g}" if f.value is not None else "")
                for f in facts
            )
            or "(none)"
        )
        verdicts: list[CheckedVerdict] = []
        for group in self.rulebook.groups:
            rules = self.rulebook.rules_in(group.id)
            system, user = self.assessment_prompt.render(
                organisation=self.rulebook.organisation,
                group=group.title,
                rules="\n".join(_rule_line(r) for r in rules),
                facts=facts_text,
                call_id=transcript.call_id,
                transcript=text,
            )
            response = self._call(
                run_id,
                transcript.call_id,
                f"assessment:{group.id}",
                LLMRequest(
                    task="assessment",
                    model=self.settings.assessment_model,
                    system=system,
                    user=user,
                    json_schema=assessment_schema([r.id for r in rules]),
                    payload={
                        "call_id": transcript.call_id,
                        "turns": turns,
                        "facts": [f.model_dump() for f in facts],
                        "rules": [json.loads(r.model_dump_json()) for r in rules],
                    },
                ),
            )
            try:
                if response is None:
                    raise LLMError("assessment call failed")
                raw = Assessment.model_validate_json(response.text).verdicts
            except (LLMError, ValidationError):
                errors += 1
                verdicts.extend(failed_verdicts(rules, f"Assessment failed for {group.id}."))
                continue
            verdicts.extend(
                check_verdicts(raw, rules, transcript, self.settings.confidence_threshold)
            )
        return facts, verdicts, dropped, errors
