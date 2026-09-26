"""SQLite persistence for runs and episodes (traces stored as JSON per episode)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.trace import Episode, StepRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    run_id            TEXT NOT NULL REFERENCES runs(id),
    agent             TEXT NOT NULL,
    task              TEXT NOT NULL,
    seed              INTEGER NOT NULL,
    success           INTEGER NOT NULL,
    steps             INTEGER NOT NULL,
    total_return      REAL NOT NULL,
    truncated         INTEGER NOT NULL,
    invalid_actions   INTEGER NOT NULL,
    duration_ms       REAL NOT NULL,
    final_observation TEXT NOT NULL,
    trace_json        TEXT NOT NULL,
    PRIMARY KEY (run_id, agent, task, seed)
);
"""


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------- runs

    def ensure_run(self, run_id: str, config: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO runs (id, created_at, config_json) VALUES (?, ?, ?)",
            (run_id, datetime.now(UTC).isoformat(timespec="seconds"), json.dumps(config)),
        )
        self.conn.commit()

    def runs(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM runs ORDER BY created_at DESC, id").fetchall()
        return [dict(r) for r in rows]

    def latest_run_id(self) -> str | None:
        row = self.conn.execute(
            "SELECT id FROM runs ORDER BY created_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
        return None if row is None else str(row["id"])

    # --------------------------------------------------------------- episodes

    def has_episode(self, run_id: str, agent: str, task: str, seed: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM episodes WHERE run_id=? AND agent=? AND task=? AND seed=?",
            (run_id, agent, task, seed),
        ).fetchone()
        return row is not None

    def save_episode(self, ep: Episode) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ep.run_id,
                ep.agent,
                ep.task,
                ep.seed,
                int(ep.success),
                ep.steps,
                ep.total_return,
                int(ep.truncated),
                ep.invalid_actions,
                ep.duration_ms,
                ep.final_observation,
                json.dumps(ep.trace_dicts()),
            ),
        )
        self.conn.commit()

    def episodes(self, run_id: str, with_trace: bool = False) -> list[Episode]:
        rows = self.conn.execute(
            "SELECT * FROM episodes WHERE run_id=? ORDER BY agent, task, seed", (run_id,)
        ).fetchall()
        return [_to_episode(r, with_trace) for r in rows]

    def episode(self, run_id: str, agent: str, task: str, seed: int) -> Episode | None:
        row = self.conn.execute(
            "SELECT * FROM episodes WHERE run_id=? AND agent=? AND task=? AND seed=?",
            (run_id, agent, task, seed),
        ).fetchone()
        return None if row is None else _to_episode(row, with_trace=True)


def _to_episode(row: sqlite3.Row, with_trace: bool) -> Episode:
    trace = [StepRecord(**d) for d in json.loads(row["trace_json"])] if with_trace else []
    return Episode(
        run_id=row["run_id"],
        agent=row["agent"],
        task=row["task"],
        seed=row["seed"],
        success=bool(row["success"]),
        steps=row["steps"],
        total_return=row["total_return"],
        truncated=bool(row["truncated"]),
        invalid_actions=row["invalid_actions"],
        duration_ms=row["duration_ms"],
        final_observation=row["final_observation"],
        trace=trace,
    )
