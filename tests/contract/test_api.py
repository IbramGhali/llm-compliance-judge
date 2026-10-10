from __future__ import annotations

from fastapi.testclient import TestClient

from compliance_judge import synth
from compliance_judge.api import create_app
from compliance_judge.app import App


def _client(small_app: App) -> tuple[TestClient, str]:
    transcripts = synth.load_transcripts(small_app.settings.data_dir)
    small_app.store.ingest(transcripts)
    run_id = small_app.pipeline.run(transcripts).run_id
    return TestClient(create_app(small_app.settings)), run_id


def test_health_and_runs(small_app: App) -> None:
    client, run_id = _client(small_app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/runs").json()[0]["run_id"] == run_id


def test_verdicts_filter_and_unknown_run(small_app: App) -> None:
    client, run_id = _client(small_app)
    breaches = client.get(f"/runs/{run_id}/verdicts", params={"verdict": "breach"}).json()
    assert breaches and all(v["verdict"] == "breach" for v in breaches)
    assert client.get("/runs/nope/verdicts").status_code == 404


def test_review_endpoint_validates_and_records(small_app: App) -> None:
    client, run_id = _client(small_app)
    item = client.get(f"/runs/{run_id}/queue").json()[0]
    bad = client.post(f"/verdicts/{item['id']}/review", json={"decision": "maybe", "reviewer": "r"})
    assert bad.status_code == 422
    label = None if item["verdict"] == "breach" else "pass"
    ok = client.post(
        f"/verdicts/{item['id']}/review",
        json={"decision": "confirm", "reviewer": "r", "label": label},
    )
    assert ok.status_code == 201
    assert (
        client.post(
            "/verdicts/999999/review", json={"decision": "confirm", "reviewer": "r"}
        ).status_code
        == 404
    )


def test_reports_and_reviewer_page(small_app: App) -> None:
    client, _ = _client(small_app)
    assert len(client.get("/reports/rule-health").json()) == 16
    assert client.get("/reports/agents").json()
    page = client.get("/")
    assert page.status_code == 200 and "Review queue" in page.text
