# ADR 0002: Rules and prompts are versioned data, and every verdict records its versions

Status: accepted

## Context

Compliance rules change often: a new disclosure is required, a threshold moves, wording is clarified after a dispute. If rules live in code or inside one prompt, every change is a deploy, and an old verdict cannot be explained because nobody knows which wording produced it.

## Options

1. Rules hard-coded in the prompt template.
2. Rules in a database edited through an admin screen.
3. Rules and prompts as files under version control, with explicit versions, validated on load.

## Decision

Option 3 for this repository. `config/rulebook.yaml` holds every rule with an id, a per-rule version, a kind (`required`, `prohibited`, `ordering`, `threshold`, `conditional`) and the fact labels it depends on. Prompts are Markdown files with a version in front matter. Loading fails fast on unknown labels, missing kind-specific fields or duplicate ids.

Each run stores the rulebook version, both prompt versions and both model ids; each verdict stores its rule version.

## Consequences

- A rule change is a reviewed pull request, not a deploy, and CI re-runs the golden set against it.
- Any verdict can be traced to the exact rule wording, prompt and model that produced it.
- Rule health can compare fire rates across rule versions to spot a change that made a rule noisier.
- Accepted cost: non-engineers cannot edit rules without Git. The production version of this design moves the same schema into a database table with an approval workflow; the validation and versioning stay the same.
