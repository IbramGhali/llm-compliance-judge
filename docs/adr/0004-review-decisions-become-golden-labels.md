# ADR 0004: Human review decisions become golden-set labels

Status: accepted

## Context

An evaluation set built once goes stale. New phrasing, new products and new rules appear, and the cases that matter most are the ones where the judge and a human disagree. Reviewers already look at flagged verdicts; their decisions are the most valuable labels available.

## Options

1. A fixed golden set, refreshed by hand occasionally.
2. Log review decisions for reporting only.
3. Feed review decisions back into the golden set, with the source recorded, and gate CI on that set.

## Decision

Option 3. A reviewer confirms or dismisses each breach, and labels each `unclear` verdict. `judge review export` merges those decisions into `golden/labels.jsonl` with `source: review`, replacing any synthetic label for the same call and rule. CI evaluates against the golden set and fails if macro F1 drops more than 2 points below `eval/baseline.json`.

## Consequences

- The golden set grows where the judge is weakest, so the regression gate protects the cases that actually went wrong.
- Reviewer disagreement per rule is reported in rule health; a rule that reviewers keep dismissing is a candidate for rewording.
- Accepted cost: reviewer labels can be wrong or inconsistent. Labels keep their source and an optional note, so a disputed label can be found and corrected, and a second-reviewer step is on the roadmap.
