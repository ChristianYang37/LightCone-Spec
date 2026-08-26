"""SQLite-backed job state and crash recovery."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .protocol import Job

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    node TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed','skipped')),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    assigned_gpus TEXT,
    started_at TEXT,
    completed_at TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS jobs_node_status ON jobs(node, status, ordinal);
CREATE TABLE IF NOT EXISTS attempts (
    job_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    error TEXT,
    PRIMARY KEY(job_id, attempt),
    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
);
CREATE TABLE IF NOT EXISTS stage_state (
    node TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS selections (
    name TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class StateStore:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "state.sqlite"
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=60)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def recover_interrupted(self) -> int:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT job_id, attempt_count FROM jobs WHERE status='running'"
            ).fetchall()
            for row in rows:
                connection.execute(
                    "UPDATE attempts SET status='interrupted', completed_at=CURRENT_TIMESTAMP, error='runner interrupted' WHERE job_id=? AND attempt=? AND status='running'",
                    (row["job_id"], row["attempt_count"]),
                )
            connection.execute(
                "UPDATE jobs SET status='pending', assigned_gpus=NULL, started_at=NULL, error='previous runner interrupted' WHERE status='running'"
            )
            return len(rows)

    def add_jobs(self, node: str, jobs: tuple[Job, ...]) -> None:
        with self.connect() as connection:
            existing_rows = connection.execute(
                "SELECT config_json FROM jobs WHERE node=? ORDER BY ordinal", (node,)
            ).fetchall()
            existing = len(existing_rows)
            if existing > len(jobs):
                raise RuntimeError(f"node {node} cannot shrink from {existing} to {len(jobs)} rows")
            for index, row in enumerate(existing_rows):
                expected = json.dumps(jobs[index].to_dict(), sort_keys=True)
                if row["config_json"] != expected:
                    raise RuntimeError(f"node {node} row {index} changed after materialization")
            for job in jobs:
                connection.execute(
                    "INSERT OR IGNORE INTO jobs(job_id,node,ordinal,config_json,status) VALUES(?,?,?,?, 'pending')",
                    (job.job_id, node, job.ordinal, json.dumps(job.to_dict(), sort_keys=True)),
                )
            connection.execute(
                "INSERT INTO stage_state(node,status,row_count) VALUES(?, 'pending', ?) "
                "ON CONFLICT(node) DO UPDATE SET "
                "status=CASE WHEN excluded.row_count > stage_state.row_count THEN 'pending' ELSE stage_state.status END, "
                "row_count=excluded.row_count, updated_at=CURRENT_TIMESTAMP",
                (node, len(jobs)),
            )

    def add_internal_jobs(self, jobs: tuple[Job, ...], *, storage_node: str | None = None) -> None:
        """Persist resumable calibration work without adding a paper DAG node."""
        with self.connect() as connection:
            for job in jobs:
                payload = json.dumps(job.to_dict(), sort_keys=True)
                row = connection.execute(
                    "SELECT config_json FROM jobs WHERE job_id=?", (job.job_id,)
                ).fetchone()
                if row is not None and row["config_json"] != payload:
                    raise RuntimeError(f"internal job {job.job_id} changed")
                connection.execute(
                    "INSERT OR IGNORE INTO jobs(job_id,node,ordinal,config_json,status) "
                    "VALUES(?,?,?,?, 'pending')",
                    (job.job_id, storage_node or job.node, job.ordinal, payload),
                )

    def completed_attempt_dir(self, job_id: str) -> Path | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT output_dir FROM attempts WHERE job_id=? AND status='completed' "
                "ORDER BY attempt DESC LIMIT 1",
                (job_id,),
            ).fetchone()
        return None if row is None else Path(row["output_dir"])

    def pending_jobs(self, node: str) -> tuple[Job, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT config_json FROM jobs WHERE node=? AND status='pending' ORDER BY ordinal",
                (node,),
            ).fetchall()
        return tuple(Job(**json.loads(row["config_json"])) for row in rows)

    def jobs(self, node: str) -> tuple[Job, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT config_json FROM jobs WHERE node=? ORDER BY ordinal", (node,)
            ).fetchall()
        return tuple(Job(**json.loads(row["config_json"])) for row in rows)

    def start(self, job: Job, gpus: tuple[int, ...], output_dir: Path) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT status, attempt_count FROM jobs WHERE job_id=?", (job.job_id,)
            ).fetchone()
            if row is None or row["status"] != "pending":
                raise RuntimeError(f"job {job.job_id} is not pending")
            attempt = int(row["attempt_count"]) + 1
            changed = connection.execute(
                "UPDATE jobs SET status='running', attempt_count=?, assigned_gpus=?, started_at=CURRENT_TIMESTAMP, error=NULL WHERE job_id=? AND status='pending'",
                (attempt, ",".join(map(str, gpus)), job.job_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError(f"job {job.job_id} was claimed concurrently")
            connection.execute(
                "INSERT INTO attempts(job_id,attempt,status,output_dir) VALUES(?,?, 'running', ?)",
                (job.job_id, attempt, str(output_dir)),
            )
            return attempt

    def next_attempt(self, job_id: str) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT attempt_count FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return int(row["attempt_count"]) + 1

    def failed_attempts(self, job_id: str) -> int:
        with self.connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM attempts WHERE job_id=? AND status='failed'",
                    (job_id,),
                ).fetchone()[0]
            )

    def skip_job(self, job_id: str, reason: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE jobs SET status='skipped', completed_at=CURRENT_TIMESTAMP, error=? "
                "WHERE job_id=? AND status IN ('pending','running')",
                (reason, job_id),
            )

    def complete(self, job_id: str, attempt: int) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE attempts SET status='completed', completed_at=CURRENT_TIMESTAMP WHERE job_id=? AND attempt=?",
                (job_id, attempt),
            )
            connection.execute(
                "UPDATE jobs SET status='completed', completed_at=CURRENT_TIMESTAMP, error=NULL WHERE job_id=?",
                (job_id,),
            )

    def fail(self, job_id: str, attempt: int, error: str, *, retry: bool) -> None:
        status = "pending" if retry else "failed"
        with self.connect() as connection:
            connection.execute(
                "UPDATE attempts SET status='failed', completed_at=CURRENT_TIMESTAMP, error=? WHERE job_id=? AND attempt=?",
                (error, job_id, attempt),
            )
            connection.execute(
                "UPDATE jobs SET status=?, completed_at=CURRENT_TIMESTAMP, assigned_gpus=NULL, error=? WHERE job_id=?",
                (status, error, job_id),
            )

    def skip_pending(self, node: str, reason: str) -> int:
        with self.connect() as connection:
            changed = connection.execute(
                "UPDATE jobs SET status='skipped', completed_at=CURRENT_TIMESTAMP, error=? WHERE node=? AND status='pending'",
                (reason, node),
            ).rowcount
            connection.execute(
                "UPDATE stage_state SET status='skipped', updated_at=CURRENT_TIMESTAMP WHERE node=?",
                (node,),
            )
            return changed

    def finish_stage(self, node: str) -> str:
        counts = self.status_counts(node)
        status = (
            "completed"
            if not counts.get("pending") and not counts.get("running") and not counts.get("failed")
            else "failed"
        )
        with self.connect() as connection:
            connection.execute(
                "UPDATE stage_state SET status=?, updated_at=CURRENT_TIMESTAMP WHERE node=?",
                (status, node),
            )
        return status

    def mark_stage_failed(self, node: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE stage_state SET status='failed', updated_at=CURRENT_TIMESTAMP WHERE node=?",
                (node,),
            )

    def status_counts(self, node: str | None = None) -> dict[str, int]:
        query = "SELECT status, COUNT(*) AS count FROM jobs"
        parameters: tuple[object, ...] = ()
        if node is not None:
            query += " WHERE node=?"
            parameters = (node,)
        query += " GROUP BY status"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def stage_rows(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT s.node,s.status,s.row_count,s.updated_at,"
                "SUM(CASE WHEN j.status='completed' THEN 1 ELSE 0 END) AS completed,"
                "SUM(CASE WHEN j.status='failed' THEN 1 ELSE 0 END) AS failed,"
                "SUM(CASE WHEN j.status='skipped' THEN 1 ELSE 0 END) AS skipped "
                "FROM stage_state s LEFT JOIN jobs j ON j.node=s.node GROUP BY s.node ORDER BY MIN(j.rowid)"
            ).fetchall()
        return [dict(row) for row in rows]

    def stage_status(self, node: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT status FROM stage_state WHERE node=?", (node,)
            ).fetchone()
        return None if row is None else str(row["status"])

    def set_selection(self, name: str, value: Any) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO selections(name,value_json) VALUES(?,?) ON CONFLICT(name) DO UPDATE SET value_json=excluded.value_json, updated_at=CURRENT_TIMESTAMP",
                (name, json.dumps(value, sort_keys=True)),
            )

    def selection(self, name: str, default: Any = None) -> Any:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM selections WHERE name=?", (name,)
            ).fetchone()
        return default if row is None else json.loads(row["value_json"])

    def completed_attempt_dirs(self, node: str) -> tuple[Path, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT a.output_dir FROM attempts a JOIN jobs j ON j.job_id=a.job_id WHERE j.node=? AND a.status='completed' ORDER BY j.ordinal",
                (node,),
            ).fetchall()
        return tuple(Path(row["output_dir"]) for row in rows)
