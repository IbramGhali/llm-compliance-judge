"""FastAPI service: JSON endpoints plus a minimal server-rendered reviewer UI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from compliance_judge import reports, review
from compliance_judge.app import App, build_app
from compliance_judge.config import Settings

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


class ReviewIn(BaseModel):
    decision: str
    reviewer: str
    label: str | None = None
    comment: str | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    state: App = build_app(settings)
    api = FastAPI(title="llm-compliance-judge", version="0.1.0")

    def latest() -> str:
        run_id = state.store.latest_run()
        if not run_id:
            raise HTTPException(404, "no completed run")
        return run_id

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "rulebook": state.rulebook.version}

    @api.get("/runs")
    def runs() -> list[dict[str, Any]]:
        return state.store.runs()

    @api.get("/runs/{run_id}/verdicts")
    def verdicts(run_id: str, verdict: str | None = None) -> list[dict[str, Any]]:
        if state.store.run(run_id) is None:
            raise HTTPException(404, "run not found")
        rows = state.store.verdicts(run_id)
        return [r for r in rows if verdict is None or r["verdict"] == verdict]

    @api.get("/runs/{run_id}/queue")
    def queue(run_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return state.store.review_queue(run_id, limit)

    @api.post("/verdicts/{verdict_id}/review", status_code=201)
    def post_review(verdict_id: int, body: ReviewIn) -> dict[str, str]:
        try:
            review.decide(
                state.store, verdict_id, body.decision, body.reviewer, body.label, body.comment
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"status": "recorded"}

    @api.get("/reports/rule-health")
    def rule_health(run_id: str | None = None) -> list[dict[str, Any]]:
        return [h.__dict__ for h in reports.rule_health(state.store, run_id or latest())]

    @api.get("/reports/agents")
    def agents(run_id: str | None = None) -> list[dict[str, Any]]:
        return reports.agent_scorecard(state.store, run_id or latest())

    @api.get("/", response_class=HTMLResponse)
    def review_page(request: Request) -> HTMLResponse:
        run_id = state.store.latest_run()
        items = []
        for v in state.store.review_queue(run_id, 25) if run_id else []:
            transcript = state.store.transcript(v["call_id"])
            turn = None
            if transcript and v["turn_index"] is not None:
                turn = next((t for t in transcript.turns if t.index == v["turn_index"]), None)
            items.append({**v, "rule": state.rulebook.get(v["rule_id"]), "turn": turn})
        return TEMPLATES.TemplateResponse(
            request, "review.html", {"run_id": run_id, "items": items}
        )

    @api.post("/ui/review/{verdict_id}")
    def ui_review(
        verdict_id: int,
        decision: Annotated[str, Form()],
        label: Annotated[str | None, Form()] = None,
    ) -> RedirectResponse:
        try:
            review.decide(state.store, verdict_id, decision, "ui-reviewer", label or None)
        except (KeyError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return RedirectResponse("/", status_code=303)

    return api
