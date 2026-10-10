"""Domain types shared by every stage. Pydantic models double as the JSON contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Speaker = Literal["agent", "customer"]


class Turn(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    speaker: Speaker
    t: float = Field(description="Seconds from the start of the call")
    text: str


class Transcript(BaseModel):
    call_id: str
    agent: str
    customer: str
    course: str
    started_at: str
    turns: list[Turn]

    def full_text(self) -> str:
        return "\n".join(turn.text for turn in self.turns)


# Fact labels the extraction stage may emit. Kept as data so the prompt, the schema and
# the rulebook validation all read from one list.
FACT_LABELS: dict[str, str] = {
    "recording_notice": "Agent says the call is recorded",
    "identity_confirmed": "Agent confirms the customer's identity (name, date of birth)",
    "fees_discussed": "Any price, fee or dollar amount for the course is mentioned",
    "total_price_disclosed": "Agent states the total price of the whole course",
    "payment_plan_offered": "Agent offers to spread payments over time",
    "instalment_terms_stated": "Agent states instalment amount and number of instalments",
    "discount_offered": "Agent offers a discount; value is the percentage",
    "entry_requirements_stated": "Agent states the entry requirements",
    "duration_stated": "Agent states course duration or weekly study hours",
    "cooling_off_disclosed": "Agent explains the cooling-off cancellation period",
    "refund_policy_explained": "Agent explains refunds after the cooling-off period",
    "guaranteed_outcome_claim": "Agent promises a guaranteed job, pass or outcome",
    "false_urgency_claim": "Agent uses a false deadline or says an offer ends now",
    "accreditation_claim": "Agent calls the course a university degree or university accredited",
    "salary_claim": "Agent quotes a specific salary or earnings",
    "cancellation_discouraged": "Agent says the customer cannot cancel or discourages cancelling",
    "hardship_mentioned": "Customer mentions financial hardship",
    "callback_offered": "Agent offers to call back or give the customer time to think",
    "enrolment_consent_given": "Customer explicitly agrees to enrol",
    "enrolment_processed": "Agent states the enrolment has been processed",
}


class Fact(BaseModel):
    label: str
    turn_index: int
    quote: str = Field(description="Exact text copied from the transcript turn")
    value: float | None = None


class Extraction(BaseModel):
    facts: list[Fact]


class Verdict(StrEnum):
    PASS = "pass"
    BREACH = "breach"
    UNCLEAR = "unclear"


class RuleVerdict(BaseModel):
    rule_id: str
    verdict: Verdict
    evidence_quote: str | None = None
    turn_index: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class Assessment(BaseModel):
    verdicts: list[RuleVerdict]


class CheckedVerdict(RuleVerdict):
    """A verdict after deterministic post-checks. `raw_verdict` is what the model said."""

    raw_verdict: Verdict
    rule_version: int
    post_check: str = "ok"


class LabelSource(StrEnum):
    SYNTHETIC = "synthetic"
    REVIEW = "review"


class GoldenLabel(BaseModel):
    call_id: str
    rule_id: str
    label: Literal["pass", "breach"]
    source: LabelSource = LabelSource.SYNTHETIC
    note: str | None = None
