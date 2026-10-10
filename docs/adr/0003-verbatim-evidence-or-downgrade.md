# ADR 0003: A breach needs a verbatim quote, or it is downgraded to unclear

Status: accepted

## Context

An LLM judge that says "breach" must show where. Models sometimes paraphrase quotes or produce plausible text that was never said. A compliance team acting on an invented quote is worse than no flag at all, because it damages trust in every other flag and can lead to unfair action against an agent.

## Options

1. Trust the model's quote.
2. Ask the model to double-check its own quote.
3. Check the quote in code: it must appear verbatim (whitespace-normalised, case-sensitive) inside a single transcript turn. If not, downgrade the verdict to `unclear` and send it to a human.

## Decision

Option 3, applied as a deterministic post-check to every verdict before it is stored. Breaches of `prohibited`, `ordering` and `threshold` rules must carry a quote. A `required` breach is evidence of absence and has no quote; the rationale says what was missing. Extracted facts get the same check against the turn they cite.

Verdicts with confidence below the threshold (default 0.6) and model calls that fail or return invalid JSON also become `unclear`.

## Consequences

- No unverified quote reaches a reviewer as a confirmed breach. In the offline run, 31 drifted quotes were caught: 5 breach verdicts were downgraded to `unclear`, and the rest were on pass verdicts, where they are flagged but harmless.
- Grounding rate becomes a metric (95.0% offline), so a model or prompt change that makes quoting worse shows up in evaluation.
- Accepted cost: a correct breach with a slightly mangled quote costs a human review. We prefer that to the opposite error.
