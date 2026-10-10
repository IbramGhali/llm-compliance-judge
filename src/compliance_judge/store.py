"""SQLite store. Every verdict records the run, model, prompt version and rule version that
produced it. Writes are idempotent: re-ingesting a transcript or re-running a call in the
same run updates rows instead of duplicating them."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compliance_judge.models import CheckedVerdict, Fact, Transcript

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    call_id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    course TEXT NOT NULL,
    started_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    adapter TEXT NOT NULL,
    extraction_model TEXT NOT NULL,
    assessment_model TEXT NOT NULL,
    extraction_prompt TEXT NOT NULL,
    assessment_prompt TEXT NOT NULL,
    rulebook_version TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS extractions (
    run_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    facts TEXT NOT NULL,
    PRIMARY KEY (run_id, call_id)
);
CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    rule_version INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    raw_verdict TEXT NOT NULL,
    evidence_quote TEXT,
    turn_index INTEGER,
    confidence REAL NOT NULL,
    rationale TEXT NOT NULL,
    post_check TEXT NOT NULL,
    UNIQUE (run_id, call_id, rule_id)
);
CREATE TABLE IF NOT EXISTS reviews (
    verdict_id INTEGER PRIMARY KEY REFERENCES verdicts(id),
    decision TEXT NOT NULL CHECK (decision IN ('confirm', 'dismiss')),
    label TEXT NOT NULL CHECK (label IN ('pass', 'breach')),
    reviewer TEXT NOT NULL,
    comment TEXT,
    decided_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_calls (
    run_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    ok INTEGER NOT NULL,
    PRIMARY KEY (run_id, call_id, stage)
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self.conn:
            yield self.conn

    # transcripts -----------------------------------------------------------------------
    def ingest(self, transcripts: list[Transcript]) -> tuple[int, int]:
        """Insert new transcripts, update changed ones. Returns (added_or_changed, unchanged)."""
        changed = unchanged = 0
        with self.tx() as c:
            for t in transcripts:
                body = t.model_dump_json()
                digest = hashlib.sha256(body.encode()).hexdigest()
                row = c.execute(
                    "SELECT content_hash FROM transcripts WHERE call_id = ?", (t.call_id,)
                ).fetchone()
                if row and row["content_hash"] == digest:
                    unchanged += 1
                    continue
                c.execute(
                    "INSERT INTO transcripts VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(call_id) DO "
                    "UPDATE SET agent=excluded.agent, course=excluded.course, "
                    "started_at=excluded.started_at, content_hash=excluded.content_hash, "
                    "body=excluded.body",
                    (t.call_id, t.agent, t.course, t.started_at, digest, body),
                )
                changed += 1
        return changed, unchanged

    def transcripts(self, call_ids: list[str] | None = None) -> list[Transcript]:
        rows = self.conn.execute("SELECT body FROM transcripts ORDER BY call_id").fetchall()
        items = [Transcript.model_validate_json(r["body"]) for r in rows]
        if call_ids is not None:
            wanted = set(call_ids)
            items = [t for t in items if t.call_id in wanted]
        return items

    def transcript(self, call_id: str) -> Transcript | None:
        row = self.conn.execute(
            "SELECT body FROM transcripts WHERE call_id = ?", (call_id,)
        ).fetchone()
        return Transcript.model_validate_json(row["body"]) if row else None

    # runs ------------------------------------------------------------------------------
    def create_run(self, **meta: str) -> str:
        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        with self.tx() as c:
            c.execute(
                "INSERT INTO runs VALUES (:run_id, :created_at, :adapter, :extraction_model, "
                ":assessment_model, :extraction_prompt, :assessment_prompt, "
                ":rulebook_version, 'running')",
                {"run_id": run_id, "created_at": _now(), **meta},
            )
        return run_id

    def finish_run(self, run_id: str, status: str = "complete") -> None:
        with self.tx() as c:
            c.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))

    def runs(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM runs ORDER BY created_at DESC, run_id DESC")
        return [dict(r) for r in rows]

    def latest_run(self) -> str | None:
        row = self.conn.execute(
            "SELECT run_id FROM runs WHERE status = 'complete' "
            "ORDER BY created_at DESC, run_id DESC LIMIT 1"
        ).fetchone()
        return str(row["run_id"]) if row else None

    def run(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def done_calls(self, run_id: str) -> set[str]:
        rows = self.conn.execute("SELECT call_id FROM extractions WHERE run_id = ?", (run_id,))
        return {r["call_id"] for r in rows}

    # results ---------------------------------------------------------------------------
    def save_call_result(
        self, run_id: str, call_id: str, facts: list[Fact], verdicts: list[CheckedVerdict]
    ) -> None:
        with self.tx() as c:
            for v in verdicts:
                c.execute(
                    "INSERT INTO verdicts (run_id, call_id, rule_id, rule_version, verdict, "
                    "raw_verdict, evidence_quote, turn_index, confidence, rationale, post_check) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(run_id, call_id, rule_id) DO UPDATE SET "
                    "rule_version=excluded.rule_version, verdict=excluded.verdict, "
                    "raw_verdict=excluded.raw_verdict, evidence_quote=excluded.evidence_quote, "
                    "turn_index=excluded.turn_index, confidence=excluded.confidence, "
                    "rationale=excluded.rationale, post_check=excluded.post_check",
                    (
                        run_id,
                        call_id,
                        v.rule_id,
                        v.rule_version,
                        v.verdict.value,
                        v.raw_verdict.value,
                        v.evidence_quote,
                        v.turn_index,
                        v.confidence,
                        v.rationale,
                        v.post_check,
                    ),
                )
            # Written last: a call counts as done only once its verdicts are stored.
            c.execute(
                "INSERT INTO extractions VALUES (?, ?, ?) ON CONFLICT(run_id, call_id) "
                "DO UPDATE SET facts=excluded.facts",
                (run_id, call_id, json.dumps([f.model_dump() for f in facts])),
            )

    def log_llm_call(
        self,
        run_id: str,
        call_id: str,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        ok: bool,
    ) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id, call_id, stage) DO UPDATE SET model=excluded.model, "
                "input_tokens=excluded.input_tokens, output_tokens=excluded.output_tokens, "
                "latency_ms=excluded.latency_ms, ok=excluded.ok",
                (run_id, call_id, stage, model, input_tokens, output_tokens, latency_ms, int(ok)),
            )

    def verdicts(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT v.*, t.agent FROM verdicts v JOIN transcripts t USING (call_id) "
            "WHERE run_id = ? ORDER BY call_id, rule_id",
            (run_id,),
        )
        return [dict(r) for r in rows]

    def verdict(self, verdict_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM verdicts WHERE id = ?", (verdict_id,)).fetchone()
        return dict(row) if row else None

    def llm_calls(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM llm_calls WHERE run_id = ?", (run_id,))
        return [dict(r) for r in rows]

    # reviews ---------------------------------------------------------------------------
    def review_queue(self, run_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT v.* FROM verdicts v LEFT JOIN reviews r ON r.verdict_id = v.id "
            "WHERE v.run_id = ? AND v.verdict IN ('breach', 'unclear') AND r.verdict_id IS NULL "
            "ORDER BY CASE v.verdict WHEN 'unclear' THEN 0 ELSE 1 END, v.confidence, v.id "
            "LIMIT ?",
            (run_id, limit),
        )
        return [dict(r) for r in rows]

    def record_review(
        self, verdict_id: int, decision: str, label: str, reviewer: str, comment: str | None = None
    ) -> None:
        if self.verdict(verdict_id) is None:
            raise KeyError(f"verdict {verdict_id} not found")
        with self.tx() as c:
            c.execute(
                "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(verdict_id) DO UPDATE "
                "SET decision=excluded.decision, label=excluded.label, "
                "reviewer=excluded.reviewer, comment=excluded.comment, "
                "decided_at=excluded.decided_at",
                (verdict_id, decision, label, reviewer, comment, _now()),
            )

    def reviews(self, run_id: str | None = None) -> list[dict[str, Any]]:
        sql = (
            "SELECT r.*, v.run_id, v.call_id, v.rule_id, v.verdict FROM reviews r "
            "JOIN verdicts v ON v.id = r.verdict_id"
        )
        params: tuple[str, ...] = ()
        if run_id:
            sql += " WHERE v.run_id = ?"
            params = (run_id,)
        return [dict(r) for r in self.conn.execute(sql + " ORDER BY r.verdict_id", params)]
