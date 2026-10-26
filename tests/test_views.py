import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from harness.cli import main
from harness.envs import make_env
from harness.runner import run_matrix
from harness.server import create_app
from harness.store import Store
from harness.views import (
    agent_slug,
    episode_doc,
    episode_path,
    export,
    leaderboard_doc,
    parse_frame,
    runs_doc,
)

TASKS = ["empty-5x5", "lavagap-5x5"]


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "runs.db"
    with Store(path) as store:
        run_matrix(store, "r", ["random", "scripted"], TASKS, range(6))
    return path


def test_parse_frame_recovers_the_agent_and_grid() -> None:
    env = make_env("doorkey-5x5")
    obs = env.reset(0)
    frame = parse_frame(obs.text)
    assert frame is not None
    grid, (ax, ay) = obs.state["grid"], obs.state["agent"]
    assert frame["grid"] == grid  # the agent's glyph is replaced by the floor it stands on
    assert frame["agent"] == [ax, ay] and frame["direction"] == obs.state["direction"]
    assert frame["carrying"] is False
    assert parse_frame("no map here") is None


def test_leaderboard_ranks_with_paired_verdicts(db: Path) -> None:
    with Store(db) as store:
        doc = leaderboard_doc(store, "r")
        assert leaderboard_doc(store, "missing") is None
    assert doc is not None
    assert [s["id"] for s in doc["scopes"]] == ["all", *TASKS]
    for scope in doc["scopes"]:
        top, bottom = scope["rows"]
        assert (top["agent"], top["rank"], bottom["rank"]) == ("scripted", 1, 2)
        assert top["vs_next"]["agent"] == "random" and top["vs_next"]["ahead"]
        assert bottom["vs_next"] is None
    assert doc["scopes"][0]["rows"][0]["success_ci"] is None  # pooled: no single CI
    assert len(doc["episodes"]) == 2 * 2 * 6
    assert doc["episodes"][0]["path"].startswith("data/runs/r/episodes/")


def test_overall_scope_lists_agents_that_skipped_a_task(db: Path) -> None:
    with Store(db) as store:
        run_matrix(store, "r", ["scripted"], ["doorkey-5x5"], range(2))
        doc = leaderboard_doc(store, "r")
    assert doc is not None
    assert doc["scopes"][0]["excluded"] == ["random"]
    assert [r["agent"] for r in doc["scopes"][0]["rows"]] == ["scripted"]


def test_episode_doc_has_a_frame_per_step(db: Path) -> None:
    with Store(db) as store:
        doc = episode_doc(store, "r", "scripted", "lavagap-5x5", 1)
    assert doc is not None and doc["success"]
    assert len(doc["trace"]) == doc["steps"]
    assert all(step["frame"]["agent"] for step in doc["trace"])
    assert doc["final"]["frame"]["agent"] is not None


def test_export_writes_the_paths_the_server_serves(db: Path, tmp_path: Path) -> None:
    out = tmp_path / "site"
    with Store(db) as store:
        written = export(store, out)
        index = runs_doc(store)
    assert written == 1 + 1 + 24
    assert json.loads((out / "data/runs.json").read_text()) == index

    client = TestClient(create_app(db))
    for rel in [
        "data/runs.json",
        "data/runs/r/leaderboard.json",
        episode_path("r", "random", "empty-5x5", 3),
    ]:
        response = client.get("/" + rel)
        assert response.status_code == 200, rel
        assert response.json() == json.loads((out / rel).read_text())


def test_server_404s_and_unslugs_agents(db: Path) -> None:
    client = TestClient(create_app(db))
    assert client.get("/data/runs/nope/leaderboard.json").status_code == 404
    assert client.get("/data/runs/r/episodes/scripted/empty-5x5/99.json").status_code == 404
    assert agent_slug("react:claude-sonnet-5") == "react~claude-sonnet-5"


def test_server_serves_the_built_dashboard(db: Path, tmp_path: Path) -> None:
    web = tmp_path / "dist"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>dash</title>")
    client = TestClient(create_app(db, web))
    assert "dash" in client.get("/").text
    assert client.get("/data/runs.json").status_code == 200


def test_cli_export(db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "site"
    assert main(["--db", str(db), "export", "--out", str(out), "--latest"]) == 0
    assert "wrote 26 files" in capsys.readouterr().out
    assert (out / "data/runs/r/leaderboard.json").exists()


def test_final_frame_shows_the_lava_the_agent_died_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("harness.runner.canonical_name", lambda spec: spec)

    class Reckless:
        name = "reckless"

        def reset(self, task_id: str, seed: int) -> None:
            pass

        def act(self, observation: Any) -> Any:
            from harness.agents import Decision

            return Decision("forward")

    with Store(tmp_path / "runs.db") as store:
        # The agent starts at (1, 1) facing east; walk east until something stops it.
        run_matrix(
            store, "r", ["reckless"], ["lavagap-5x5"], range(3), agent_factory=lambda _: Reckless()
        )
        died = [e for e in store.episodes("r") if not e.success and not e.truncated]
        assert died, "expected at least one lava death walking straight east"
        doc = episode_doc(store, "r", "reckless", "lavagap-5x5", died[0].seed)
    assert doc is not None
    frame = doc["final"]["frame"]
    x, y = frame["agent"]
    assert frame["grid"][y][x] == "~" and frame["died"]
