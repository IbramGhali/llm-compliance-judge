# ADR 0001: Two stages, extraction then assessment, instead of one large prompt

Status: accepted

## Context

A sales call has to be judged against 16 rules. The obvious design is one prompt with the whole rulebook and the transcript, returning 16 verdicts. In practice that prompt grows with every rule, mixes two different jobs (finding what was said, and deciding whether it breaks a rule), and makes errors hard to trace: when a verdict is wrong you cannot tell whether the model missed the statement or misread the rule.

## Options

1. One prompt: transcript plus all rules, one call, 16 verdicts.
2. One prompt per rule: 16 calls, each focused, but 16 times the transcript tokens.
3. Two stages: one extraction call records facts (label, turn, verbatim quote), then one assessment call per rule group decides verdicts from those facts plus the transcript.

## Decision

Option 3. Extraction runs once per call on a cheap model (`claude-haiku-5-5` by default) because finding statements is high volume and low judgement. Assessment runs once per rule group (4 groups) on a stronger model (`claude-opus-5-5`), because deciding whether a disclosure was adequate is where judgement is needed.

## Consequences

- Errors are attributable. The facts table shows whether a miss was an extraction miss or an assessment mistake, and the evaluation can be read the same way.
- Model choice is per stage, so cost follows judgement: about 5 calls and ~6,000 tokens per transcript, estimated at $0.03 on the default models.
- Facts are reusable: a new rule that depends on existing labels needs no new extraction.
- Accepted cost: a fact the extractor misses cannot be recovered by the assessor unless it rereads the transcript, which the assessment prompt allows but does not rely on. The fake adapter models this weakness on purpose (see the R03 and R14 results in the README).
