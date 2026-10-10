"""Synthetic sales-call generator for the fictional Northwind Learning.

Every call is assembled from phrase banks with seeded randomness. Breaches are planted on
purpose and written to a ground-truth file, so the true label of every (call, rule) pair is
known without anyone labelling by hand. Phrase banks deliberately include paraphrases and
near-miss wording ("I can't guarantee a job") so the judge has realistic hard cases.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from compliance_judge.models import GoldenLabel, Speaker, Transcript, Turn

RULE_IDS = [f"R{n:02d}" for n in range(1, 17)]

# (name, breach multiplier): agents differ so the scorecard has something to show.
AGENTS: list[tuple[str, float]] = [
    ("Alex", 0.4),
    ("Jordan", 0.7),
    ("Sam", 1.0),
    ("Riley", 1.0),
    ("Casey", 1.3),
    ("Morgan", 1.6),
    ("Taylor", 2.0),
    ("Jamie", 0.8),
]
COURSES: list[tuple[str, int, int]] = [
    ("Project Management Certificate", 4200, 12),
    ("Bookkeeping Certificate", 3600, 12),
    ("Digital Marketing Diploma", 6000, 18),
    ("Data Analytics Certificate", 4800, 12),
    ("Business Administration Certificate", 3900, 12),
]
CUSTOMERS = [
    "Avery",
    "Blake",
    "Charlie",
    "Dana",
    "Eden",
    "Frankie",
    "Harper",
    "Indy",
    "Kai",
    "Logan",
    "Mika",
    "Noor",
    "Parker",
    "Quinn",
    "Reese",
    "Sasha",
]
SURNAMES = ["Hart", "Lin", "Okafor", "Patel", "Rossi", "Silva", "Novak", "Kim", "Moreau"]

BASE_BREACH: dict[str, float] = {
    "R01": 0.06,
    "R02": 0.07,
    "R03": 0.08,
    "R04": 0.06,
    "R05": 0.07,
    "R06": 0.10,
    "R07": 0.06,
    "R08": 0.06,
    "R09": 0.06,
    "R10": 0.07,
    "R11": 0.05,
    "R12": 0.05,
    "R13": 0.05,
    "R14": 0.20,
    "R15": 0.45,
    "R16": 0.05,
}

RECORDING = [
    "Just so you know, this call is recorded for quality and training purposes.",
    "Before we start, I need to let you know this call is being recorded.",
    "We keep a recording of this conversation for quality checks.",
]
IDENTITY = [
    "Can I confirm your full name and date of birth, please?",
    "Before we go on, could you verify your date of birth for me?",
    "I just need to check I'm speaking with the right person, what's your date of birth?",
]
TOTAL_PRICE = [
    "The total course price is ${total:,}, including all materials.",
    "Altogether you'd pay ${total:,} for the full course.",
    "All in, that's ${total:,} for the whole program.",
]
MONTHLY_ONLY = [
    "It works out to about ${monthly} a month.",
    "Most people pay around ${monthly} a month.",
]
PLAN_OFFER = [
    "We also offer a payment plan if that helps.",
    "Would spreading the cost over monthly payments suit you better?",
]
INSTALMENTS = [
    "That would be {n} monthly instalments of ${monthly}.",
    "You'd pay ${monthly} a month over {n} months.",
    "It's split into {n_words} payments of ${monthly}.",
]
PLAN_VAGUE = ["We can sort out the payment details later.", "The plan is really flexible."]
DISCOUNT = [
    "I can take {pct}% off the total for you.",
    "I can do {pct}% off if you enrol with me.",
]
DISCOUNT_PARAPHRASE = {
    25: "I can knock a quarter off the price for you.",
    20: "I can knock a fifth off the price for you.",
}
ENTRY = [
    "The entry requirement is Year 12 or equivalent work experience.",
    "You'll need to have finished Year 12, or have two years of relevant work experience.",
    "To get in, you need Year 12 or some work background.",
]
DURATION = [
    "Most students finish in {months} months, studying about 10 hours a week.",
    "The course takes around a year part-time.",
    "Plan on roughly ten hours of study each week.",
]
COOLING_OFF = [
    "You have a 10-day cooling-off period, during which you can cancel for a full refund.",
    "If you change your mind within 10 days, you can cancel and get all your money back.",
    "There's a ten day window to back out at no cost.",
]
REFUND = [
    "After the cooling-off period, refunds are pro-rata for units not yet started.",
    "If you withdraw later, we refund the units you haven't started.",
]
BREACH_LINES: dict[str, list[str]] = {
    "R09": [
        "I can guarantee you'll get a job when you finish.",
        "You're guaranteed to pass, everyone does.",
        "This course will definitely get you hired.",
    ],
    "R10": [
        "This price is only available if you sign up today.",
        "The offer ends tonight, so we need to lock it in now.",
        "There are only two places left, you need to decide right now.",
    ],
    "R11": [
        "This is a university degree, fully accredited by the university.",
        "You'll graduate with a bachelor's degree.",
    ],
    "R12": [
        "Graduates earn $95,000 straight away.",
        "You'll be on six figures within a year.",
        "Most of our grads are pulling in ninety grand.",
    ],
    "R13": [
        "Honestly, once you're enrolled you can't really cancel.",
        "Cancelling is a hassle, most people don't bother.",
    ],
}
NEAR_MISS = [
    "I can't guarantee a job, but our careers team will help you prepare.",
    "Nobody can promise a guaranteed job outcome, but many graduates find work.",
    "There's no rush today, take the time you need.",
    "It's a nationally recognised certificate, not a university degree.",
    "Some graduates go on to a university degree later.",
    "Salaries vary a lot, so I won't quote a number.",
    "Enrolments for this intake close at the end of the month.",
]
HARDSHIP = [
    "Money's really tight at the moment, I lost my job last month.",
    "I'm not sure I can afford it, things are hard right now.",
    "Things have been rough since the redundancy.",
]
CALLBACK = [
    "Would you like me to call you back next week once you've had time to think?",
    "No pressure at all, I can call you back another time.",
]
CONSENT_ASK = "Are you happy for me to enrol you on these terms?"
CONSENT = ["Yes, I agree, please go ahead and enrol me.", "Yes, go ahead.", "Yep, sign me up."]
PROCESSED = ["Great, I've processed your enrolment now.", "That's all done, you're enrolled."]
REASONS = [
    "I want to move into a new role at work.",
    "I've been thinking about a career change for a while.",
    "My manager suggested it.",
    "I'd like to get a formal qualification.",
]
NUM_WORDS = {12: "twelve", 18: "eighteen", 6: "six"}


@dataclass
class CallPlan:
    call_id: str
    agent: str
    customer: str
    course: str
    total: int
    months: int
    enrol: bool
    payment_plan: bool
    discount: bool
    hardship: bool
    planted: set[str] = field(default_factory=set)
    near_miss: bool = False


@dataclass
class GeneratedCall:
    transcript: Transcript
    labels: dict[str, str]
    near_miss: bool


def _pick(rng: random.Random, options: list[str]) -> str:
    """Agents mostly use the scripted wording; paraphrases are less common."""
    weights = [0.7, 0.2, 0.1][: len(options)] if len(options) <= 3 else None
    return rng.choices(options, weights=weights)[0]


class _Builder:
    def __init__(self) -> None:
        self.turns: list[Turn] = []
        self.t = 0.0

    def say(self, speaker: Speaker, *sentences: str) -> None:
        text = " ".join(s for s in sentences if s)
        if not text:
            return
        self.turns.append(
            Turn(index=len(self.turns), speaker=speaker, t=round(self.t, 1), text=text)
        )
        self.t += 1.5 + len(text) / 15


def _plan(rng: random.Random, n: int) -> CallPlan:
    agent, mult = rng.choice(AGENTS)
    course, total, months = rng.choice(COURSES)
    plan = CallPlan(
        call_id=f"call-{n:04d}",
        agent=agent,
        customer=f"{rng.choice(CUSTOMERS)} {rng.choice(SURNAMES)}",
        course=course,
        total=total,
        months=months,
        enrol=rng.random() < 0.7,
        payment_plan=rng.random() < 0.45,
        discount=rng.random() < 0.35,
        hardship=rng.random() < 0.12,
    )
    for rule_id in RULE_IDS:
        p = BASE_BREACH[rule_id] if rule_id in ("R14", "R15") else BASE_BREACH[rule_id] * mult
        if rng.random() < min(p, 0.6):
            plan.planted.add(rule_id)
    plan.near_miss = rng.random() < 0.25
    return plan


def _truth(plan: CallPlan) -> dict[str, str]:
    applies = {
        "R03": plan.enrol,
        "R05": plan.enrol,
        "R07": plan.enrol,
        "R08": plan.enrol,
        "R13": plan.enrol,
        "R16": plan.enrol,
        "R06": plan.payment_plan,
        "R14": plan.discount,
        "R15": plan.hardship and plan.enrol,
    }
    return {
        rule_id: "breach" if rule_id in plan.planted and applies.get(rule_id, True) else "pass"
        for rule_id in RULE_IDS
    }


def _build(rng: random.Random, plan: CallPlan) -> list[Turn]:
    b = _Builder()
    first = plan.customer.split()[0]
    monthly = round(plan.total / plan.months)
    p = plan.planted
    recording = _pick(rng, RECORDING)
    recording_late = "R04" in p and rng.random() < 0.5

    b.say(
        "agent",
        f"Hi {first}, it's {plan.agent} from Northwind Learning.",
        "" if "R04" in p else recording,
        f"Thanks for your interest in the {plan.course}.",
    )
    b.say("customer", "Hi, yes, thanks for calling.")

    fee_line = (rng.choice(MONTHLY_ONLY) if "R02" in p else _pick(rng, TOTAL_PRICE)).format(
        total=plan.total, monthly=monthly
    )

    def identity_block() -> None:
        b.say("agent", _pick(rng, IDENTITY))
        b.say(
            "customer",
            f"Sure, it's {plan.customer}, born {rng.randint(1, 28)} "
            f"{rng.choice(['March', 'June', 'August', 'November'])} "
            f"{rng.randint(1970, 2003)}.",
        )
        b.say("agent", "Thanks, that's confirmed.")

    if "R01" in p:
        b.say("agent", "Before anything else, " + fee_line[0].lower() + fee_line[1:])
        b.say("customer", "Okay, good to know.")
        identity_block()
    else:
        identity_block()

    b.say("agent", "What made you interested in the course?")
    b.say("customer", rng.choice(REASONS))
    if recording_late:
        b.say("agent", recording)

    info: list[str] = []
    if not ("R08" in p and plan.enrol) and (plan.enrol or rng.random() < 0.7):
        info.append(_pick(rng, DURATION).format(months=plan.months))
    if not ("R07" in p and plan.enrol) and (plan.enrol or rng.random() < 0.7):
        info.append(_pick(rng, ENTRY))
    b.say("agent", *info)
    b.say("customer", "That sounds manageable.")

    for rule_id in ("R09", "R11", "R12"):
        if rule_id in p:
            b.say("agent", rng.choice(BREACH_LINES[rule_id]))
            b.say("customer", "Oh, really?")
    if plan.near_miss:
        b.say("agent", rng.choice(NEAR_MISS))
        b.say("customer", "Fair enough.")

    if "R01" not in p:
        b.say("agent", fee_line)
    else:
        b.say("agent", "As I mentioned, that's the fee.")
    b.say("customer", "Right.")

    if plan.payment_plan:
        terms = (
            rng.choice(PLAN_VAGUE)
            if "R06" in p
            else _pick(rng, INSTALMENTS).format(
                n=plan.months, monthly=monthly, n_words=NUM_WORDS[plan.months]
            )
        )
        b.say("agent", rng.choice(PLAN_OFFER), terms)
        b.say("customer", "That could work.")

    if plan.discount:
        pct = rng.choice([20, 25]) if "R14" in p else rng.choice([5, 10, 15])
        if pct in DISCOUNT_PARAPHRASE and rng.random() < 0.35:
            b.say("agent", DISCOUNT_PARAPHRASE[pct])
        else:
            b.say("agent", rng.choice(DISCOUNT).format(pct=pct))
        b.say("customer", "That helps, thanks.")

    if "R10" in p:
        b.say("agent", rng.choice(BREACH_LINES["R10"]))
        b.say("customer", "Hmm, okay.")

    if plan.hardship:
        b.say("customer", rng.choice(HARDSHIP))
        if not ("R15" in p and plan.enrol):
            b.say("agent", rng.choice(CALLBACK))
            b.say(
                "customer",
                "Thanks, I'm alright to keep talking about it though."
                if plan.enrol
                else "Yes, please do that.",
            )

    if plan.enrol:
        if "R03" not in p:
            b.say("agent", _pick(rng, COOLING_OFF))
        if "R13" in p:
            b.say("agent", rng.choice(BREACH_LINES["R13"]))
        if "R05" not in p:
            b.say("agent", _pick(rng, REFUND))
        b.say("customer", "Okay.")
        if "R16" in p:
            if rng.random() < 0.5:
                b.say("agent", "I'll just get that processed for you.", rng.choice(PROCESSED))
                b.say("customer", "Oh, okay.")
            else:
                b.say("agent", rng.choice(PROCESSED), CONSENT_ASK)
                b.say("customer", rng.choice(CONSENT))
        else:
            b.say("agent", CONSENT_ASK)
            b.say("customer", rng.choice(CONSENT))
            b.say("agent", rng.choice(PROCESSED))
        b.say("agent", "You'll get a welcome email within the hour.")
    else:
        b.say("customer", "I'd like to think about it first.")
        b.say("agent", "Of course, take your time.")
    b.say("agent", f"Thanks for your time today, {first}.")
    return b.turns


def generate(n: int = 200, seed: int = 7) -> list[GeneratedCall]:
    rng = random.Random(seed)
    start = datetime(2026, 7, 1, 9, 0)
    calls: list[GeneratedCall] = []
    for i in range(1, n + 1):
        plan = _plan(rng, i)
        turns = _build(rng, plan)
        started = start + timedelta(hours=7 * i)
        transcript = Transcript(
            call_id=plan.call_id,
            agent=plan.agent,
            customer=plan.customer,
            course=plan.course,
            started_at=started.isoformat(),
            turns=turns,
        )
        calls.append(GeneratedCall(transcript, _truth(plan), plan.near_miss))
    return calls


def select_golden(calls: list[GeneratedCall], size: int = 60, seed: int = 11) -> list[str]:
    """Pick a golden set that covers every rule's breaches and includes near-miss calls."""
    rng = random.Random(seed)
    chosen: list[str] = []
    for rule_id in RULE_IDS:
        breaches = [c.transcript.call_id for c in calls if c.labels[rule_id] == "breach"]
        rng.shuffle(breaches)
        have = sum(1 for cid in chosen if cid in breaches)
        for cid in breaches:
            if have >= 3:
                break
            if cid not in chosen:
                chosen.append(cid)
                have += 1
    near = [
        c.transcript.call_id for c in calls if c.near_miss and c.transcript.call_id not in chosen
    ]
    rng.shuffle(near)
    chosen.extend(near[:12])
    rest = [c.transcript.call_id for c in calls if c.transcript.call_id not in chosen]
    rng.shuffle(rest)
    chosen.extend(rest[: max(0, size - len(chosen))])
    return sorted(chosen[:size])


def write_dataset(calls: list[GeneratedCall], data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "transcripts.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for call in calls:
            fh.write(call.transcript.model_dump_json() + "\n")
    with (data_dir / "ground_truth.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for call in calls:
            row = {
                "call_id": call.transcript.call_id,
                "labels": call.labels,
                "near_miss": call.near_miss,
            }
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_golden(calls: list[GeneratedCall], golden_ids: list[str], path: Path) -> None:
    by_id = {c.transcript.call_id: c for c in calls}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for cid in golden_ids:
            for rule_id, label in sorted(by_id[cid].labels.items()):
                fh.write(
                    GoldenLabel(call_id=cid, rule_id=rule_id, label=label).model_dump_json()  # type: ignore[arg-type]
                    + "\n"
                )


def load_transcripts(data_dir: Path) -> list[Transcript]:
    path = data_dir / "transcripts.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [Transcript.model_validate_json(line) for line in fh if line.strip()]


def load_ground_truth(data_dir: Path) -> dict[str, dict[str, str]]:
    with (data_dir / "ground_truth.jsonl").open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {row["call_id"]: row["labels"] for row in rows}
