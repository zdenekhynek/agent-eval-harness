"""``harness serve``: the dashboard's JSON documents, live from the SQLite store.

Routes mirror the files ``harness export`` writes, so the web app fetches the
same relative paths in both modes. Each request opens its own store
connection: SQLite connections are not shared across the worker threads that
FastAPI runs sync endpoints on.

FastAPI and uvicorn are optional (``uv sync --extra serve``); the rest of
the harness never imports this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from harness import __version__, views
from harness.store import Store


def create_app(db: str | Path, web_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="agent-eval-harness", version=__version__)

    @app.get("/data/runs.json")
    def runs() -> dict[str, Any]:
        with Store(db) as store:
            return views.runs_doc(store)

    @app.get("/data/runs/{run_id}/leaderboard.json")
    def leaderboard(run_id: str) -> dict[str, Any]:
        with Store(db) as store:
            doc = views.leaderboard_doc(store, run_id)
        if doc is None:
            raise HTTPException(404, f"no episodes in run {run_id!r}")
        return doc

    @app.get("/data/runs/{run_id}/episodes/{agent}/{task}/{seed}.json")
    def episode(run_id: str, agent: str, task: str, seed: int) -> dict[str, Any]:
        with Store(db) as store:
            doc = views.episode_doc(store, run_id, agent.replace("~", ":"), task, seed)
        if doc is None:
            raise HTTPException(404, "no such episode")
        return doc

    if web_dir is not None:
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app
