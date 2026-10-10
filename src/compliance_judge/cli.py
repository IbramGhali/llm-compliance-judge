"""Command line: `judge generate`, `judge run`, `judge eval`, `judge review ...`,
`judge report ...`, `judge serve`."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from compliance_judge import evaluation, reports, review, synth
from compliance_judge.app import App, build_app
from compliance_judge.config import Settings

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="LLM-as-a-judge compliance review for sales calls.",
)
review_app = typer.Typer(no_args_is_help=True, help="Human review queue.")
report_app = typer.Typer(no_args_is_help=True, help="Operational reports.")
app.add_typer(review_app, name="review")
app.add_typer(report_app, name="report")
console = Console()


def _app() -> App:
    return build_app(Settings.from_env())


def _run_id(a: App, run_id: str | None) -> str:
    resolved = run_id or a.store.latest_run()
    if not resolved:
        console.print("[red]No completed run. Run `judge run` first.[/red]")
        raise typer.Exit(1)
    return resolved


@app.command()
def generate(
    n: Annotated[int, typer.Option(help="Number of calls")] = 200,
    seed: Annotated[int, typer.Option(help="Random seed")] = 7,
    reset_golden: Annotated[bool, typer.Option(help="Rebuild the golden set")] = False,
) -> None:
    """Generate synthetic transcripts and ground truth. Creates the golden set if missing."""
    settings = Settings.from_env()
    calls = synth.generate(n=n, seed=seed)
    synth.write_dataset(calls, settings.data_dir)
    breaches = sum(1 for c in calls for lab in c.labels.values() if lab == "breach")
    console.print(
        f"Wrote {len(calls)} transcripts with {breaches} planted breaches to {settings.data_dir}"
    )
    if reset_golden or not settings.golden_path.is_file():
        golden_ids = synth.select_golden(calls)
        synth.write_golden(calls, golden_ids, settings.golden_path)
        console.print(f"Wrote golden set: {len(golden_ids)} calls to {settings.golden_path}")


@app.command()
def run(
    limit: Annotated[int | None, typer.Option(help="Only the first N calls")] = None,
    resume: Annotated[str | None, typer.Option(help="Resume an unfinished run id")] = None,
) -> None:
    """Ingest transcripts and run extraction and assessment over them."""
    a = _app()
    transcripts = synth.load_transcripts(a.settings.data_dir)
    changed, unchanged = a.store.ingest(transcripts)
    console.print(f"Ingested: {changed} new or changed, {unchanged} unchanged")
    summary = a.pipeline.run(transcripts[:limit] if limit else transcripts, run_id=resume)
    console.print(
        f"Run [bold]{summary.run_id}[/bold] ({a.pipeline.llm.name} adapter): "
        f"{summary.calls} calls in {summary.seconds:.1f}s, "
        f"{summary.skipped} skipped"
    )
    console.print(f"Verdicts: {dict(summary.verdicts)}")
    console.print(f"Post-check downgrades and flags: {dict(summary.post_checks)}")
    console.print(
        f"Facts dropped (quote not in cited turn): {summary.facts_dropped}; "
        f"LLM errors: {summary.llm_errors}"
    )


@app.command("eval")
def eval_cmd(
    run_id: Annotated[str | None, typer.Option("--run", help="Run id (default latest)")] = None,
    gate: Annotated[bool, typer.Option(help="Exit 1 if macro F1 regresses")] = False,
    update_baseline: Annotated[bool, typer.Option(help="Write eval/baseline.json")] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Print JSON only")] = False,
) -> None:
    """Score a run against the golden set and the full synthetic ground truth."""
    a = _app()
    rid = _run_id(a, run_id)
    truth = (
        synth.load_ground_truth(a.settings.data_dir)
        if (a.settings.data_dir / "ground_truth.jsonl").is_file()
        else None
    )
    meta = a.store.run(rid) or {}
    report = evaluation.evaluate(
        rid,
        a.store.verdicts(rid),
        evaluation.load_golden(a.settings.golden_path),
        truth,
        a.store.llm_calls(rid),
        a.settings.pricing_path,
        meta.get("extraction_model", a.settings.extraction_model),
        meta.get("assessment_model", a.settings.assessment_model),
    )
    out = a.settings.baseline_path.parent / "results" / "latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(evaluation.report_json(report) + "\n", encoding="utf-8")

    if json_out:
        console.print_json(evaluation.report_json(report))
    else:
        _print_report(report, meta)
    if update_baseline:
        evaluation.write_baseline(report, a.settings.baseline_path)
        console.print(f"Baseline written to {a.settings.baseline_path}")
    if gate:
        ok, message = evaluation.check_gate(
            report, a.settings.baseline_path, a.settings.gate_tolerance
        )
        console.print(f"[{'green' if ok else 'red'}]{message}[/]")
        if not ok:
            raise typer.Exit(1)


def _print_report(report: evaluation.EvalReport, meta: dict[str, object]) -> None:
    console.print(
        f"Run {report.run_id}: adapter {meta.get('adapter')}, rulebook "
        f"{meta.get('rulebook_version')}, prompts {meta.get('extraction_prompt')} / "
        f"{meta.get('assessment_prompt')}"
    )
    table = Table(title="Per-rule results on the golden set")
    for col in ("Rule", "Precision", "Recall", "F1", "TP", "FP", "FN", "Unclear"):
        table.add_column(col, justify="right" if col != "Rule" else "left")
    for rid, s in sorted(report.golden.rules.items()):
        table.add_row(
            rid,
            f"{s.precision:.2f}",
            f"{s.recall:.2f}",
            f"{s.f1:.2f}",
            str(s.tp),
            str(s.fp),
            str(s.fn),
            str(s.unclear),
        )
    console.print(table)
    summary = Table(title="Summary")
    for col in ("Set", "Calls", "Pairs", "Macro P", "Macro R", "Macro F1", "Kappa", "Unclear"):
        summary.add_column(col, justify="right" if col != "Set" else "left")
    for result in (report.golden, report.full):
        if result:
            summary.add_row(
                result.name,
                str(result.calls),
                str(result.pairs),
                f"{result.macro_precision:.3f}",
                f"{result.macro_recall:.3f}",
                f"{result.macro_f1:.3f}",
                f"{result.kappa:.3f}",
                f"{result.unclear_rate:.1%}",
            )
    console.print(summary)
    console.print(
        f"Evidence grounding: {report.grounding_rate:.1%} of "
        f"{report.quoted_breaches} quoted breach verdicts had a verbatim quote"
    )
    if report.cost:
        c = report.cost
        console.print(
            f"Per transcript: {c['llm_calls_per_transcript']:.0f} LLM calls, "
            f"~{c['input_tokens_per_transcript']} input and ~{c['output_tokens_per_transcript']}"
            f" output tokens, estimated ${c['cost_usd_per_transcript']:.4f} on "
            f"{c['models']['extraction']} + {c['models']['assessment']}; "
            f"pipeline latency p50 {c['latency_ms_p50']} ms, p95 {c['latency_ms_p95']} ms"
        )


@review_app.command("list")
def review_list(
    run_id: Annotated[str | None, typer.Option("--run")] = None,
    limit: Annotated[int, typer.Option()] = 20,
) -> None:
    """Show unreviewed breach and unclear verdicts, least confident first."""
    a = _app()
    table = Table(title="Review queue")
    for col in ("ID", "Call", "Rule", "Verdict", "Conf", "Check", "Evidence"):
        table.add_column(col)
    for v in a.store.review_queue(_run_id(a, run_id), limit):
        table.add_row(
            str(v["id"]),
            v["call_id"],
            v["rule_id"],
            v["verdict"],
            f"{v['confidence']:.2f}",
            v["post_check"],
            (v["evidence_quote"] or v["rationale"])[:70],
        )
    console.print(table)


@review_app.command("decide")
def review_decide(
    verdict_id: int,
    decision: Annotated[str, typer.Argument(help="confirm or dismiss")],
    label: Annotated[str | None, typer.Option(help="pass or breach (needed for unclear)")] = None,
    reviewer: Annotated[str, typer.Option()] = "reviewer",
    comment: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Confirm or dismiss a verdict."""
    a = _app()
    review.decide(a.store, verdict_id, decision, reviewer, label, comment)
    console.print(f"Recorded {decision} for verdict {verdict_id}")


@review_app.command("export")
def review_export() -> None:
    """Merge reviewer decisions into golden/labels.jsonl."""
    a = _app()
    added, replaced = review.export_to_golden(a.store, a.settings.golden_path)
    console.print(f"Golden set updated: {added} added, {replaced} replaced by reviewer labels")


@report_app.command("rules")
def report_rules(run_id: Annotated[str | None, typer.Option("--run")] = None) -> None:
    """Rule health: fire rate, reviewer disagreement and drift."""
    a = _app()
    table = Table(title="Rule health")
    for col in ("Rule", "Calls", "Fire rate", "Unclear", "Reviewed", "Disagree", "Prev", "Drift"):
        table.add_column(col)
    for h in reports.rule_health(a.store, _run_id(a, run_id)):
        table.add_row(
            h.rule_id,
            str(h.calls),
            f"{h.fire_rate:.1%}",
            f"{h.unclear_rate:.1%}",
            str(h.reviewed),
            "-" if h.disagreement_rate is None else f"{h.disagreement_rate:.0%}",
            "-" if h.previous_fire_rate is None else f"{h.previous_fire_rate:.1%}",
            "ALERT" if h.drift_alert else "",
        )
    console.print(table)


@report_app.command("agents")
def report_agents(run_id: Annotated[str | None, typer.Option("--run")] = None) -> None:
    """Per-agent scorecard."""
    a = _app()
    table = Table(title="Agent scorecard")
    for col in ("Agent", "Calls", "Breaches", "Per 10 calls", "Unclear", "Top rules"):
        table.add_column(col)
    for row in reports.agent_scorecard(a.store, _run_id(a, run_id)):
        table.add_row(
            row["agent"],
            str(row["calls"]),
            str(row["breaches"]),
            str(row["breaches_per_10_calls"]),
            str(row["unclear"]),
            row["top_rules"],
        )
    console.print(table)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the API and reviewer UI."""
    import uvicorn

    uvicorn.run("compliance_judge.api:create_app", factory=True, host=host, port=port)


if __name__ == "__main__":
    app()
