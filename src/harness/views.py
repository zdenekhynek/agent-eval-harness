"""JSON documents behind the dashboard.

The same documents are served live by ``harness serve`` and written to disk by
``harness export`` for static hosting, at the same relative paths
(``data/runs.json``, ``data/runs/<run>/leaderboard.json``,
``data/runs/<run>/episodes/<agent>/<task>/<seed>.json``), so the web app never
knows which of the two it is talking to.

A leaderboard is a list of *scopes* - all tasks pooled, then each task - and
each scope ranks agents by normalised score. Next to every rank sits a paired
comparison with the agent ranked just below it: "ahead" only when the 95% CI
of the per-seed difference excludes zero, otherwise "tied". That is what
"CI overlap shown, not just point estimates" means in practice.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from statistics import mean
from typing import Any

from harness.runner import run_references
from harness.score import Comparison, References, Score, compare, overall, score
from harness.store import Store
from harness.trace import Episode

Doc = dict[str, Any]

GLYPH_DIRECTIONS = {">": 0, "v": 1, "<": 2, "^": 3}


def agent_slug(agent: str) -> str:
    """Path-safe agent name: ``react:claude-sonnet-5`` -> ``react~claude-sonnet-5``."""
    return agent.replace(":", "~")


def episode_path(run_id: str, agent: str, task: str, seed: int) -> str:
    return f"data/runs/{run_id}/episodes/{agent_slug(agent)}/{task}/{seed}.json"


# ------------------------------------------------------------------- runs


def runs_doc(store: Store) -> Doc:
    totals = store.run_totals()
    runs = []
    for run in store.runs():
        config = run["config"]
        n, cost = totals.get(run["id"], (0, None))
        runs.append(
            {
                "id": run["id"],
                "created_at": run["created_at"],
                "agents": config.get("agents", []),
                "tasks": config.get("tasks", []),
                "episodes": n,
                "cost_usd": cost,
                "leaderboard": f"data/runs/{run['id']}/leaderboard.json",
            }
        )
    return {"runs": [r for r in runs if r["episodes"]]}


# ------------------------------------------------------------ leaderboard


def leaderboard_doc(store: Store, run_id: str) -> Doc | None:
    episodes = store.episodes(run_id)
    if not episodes:
        return None
    tasks = sorted({e.task for e in episodes})
    refs = run_references(store, run_id, tasks)
    scores = score(episodes, refs)

    scopes = [_overall_scope(episodes, scores, refs, tasks)]
    for task in tasks:
        task_eps = [e for e in episodes if e.task == task]
        rows = sorted(
            (s for s in scores if s.task == task),
            key=lambda s: (-(s.norm if s.norm is not None else -1e9), -s.success_rate, s.agent),
        )
        scopes.append(
            {
                "id": task,
                "label": task,
                "rows": _with_vs_next([_score_row(s) for s in rows], task_eps, refs, task),
                "excluded": [],
            }
        )

    return {
        "run_id": run_id,
        "tasks": tasks,
        "agents": sorted({e.agent for e in episodes}),
        "max_steps": (store.run_config(run_id) or {}).get("max_steps"),
        "references": {
            t: {"floor": r.floor, "ceiling": r.ceiling, "n": r.n} for t, r in refs.items()
        },
        "scopes": scopes,
        "episodes": [_episode_row(e, refs) for e in episodes],
    }


def _score_row(s: Score) -> Doc:
    return {
        "agent": s.agent,
        "n": s.n,
        "norm": s.norm,
        "norm_ci": list(s.norm_ci) if s.norm_ci else None,
        "success": s.success_rate,
        "success_ci": list(s.success_ci),
        "mean_return": s.mean_return,
        "steps_on_success": s.mean_steps_on_success,
        "invalid_rate": s.invalid_rate,
        "tokens": s.mean_tokens,
        "cost_usd": s.mean_cost_usd,
    }


def _overall_scope(
    episodes: list[Episode], scores: list[Score], refs: References, tasks: list[str]
) -> Doc:
    rows = []
    for o in overall(episodes, refs):
        mine = [s for s in scores if s.agent == o.agent]
        costs = [s.mean_cost_usd for s in mine if s.mean_cost_usd is not None]
        tokens = [s.mean_tokens for s in mine if s.mean_tokens is not None]
        steps = [s.mean_steps_on_success for s in mine if s.mean_steps_on_success is not None]
        rows.append(
            {
                "agent": o.agent,
                "n": o.n,
                "norm": o.norm,
                "norm_ci": list(o.norm_ci),
                # Tasks weigh equally here too, matching the normalised score.
                "success": mean(s.success_rate for s in mine),
                "success_ci": None,
                "mean_return": mean(s.mean_return for s in mine),
                "steps_on_success": mean(steps) if steps else None,
                "invalid_rate": mean(s.invalid_rate for s in mine),
                "tokens": mean(tokens) if tokens else None,
                "cost_usd": mean(costs) if costs else None,
            }
        )
    ranked = {r["agent"] for r in rows}
    return {
        "id": "all",
        "label": "All tasks",
        "rows": _with_vs_next(rows, episodes, refs, None),
        # Agents that skipped a task are not scored on an easier subset.
        "excluded": sorted({e.agent for e in episodes} - ranked),
    }


def _with_vs_next(
    rows: list[Doc], episodes: list[Episode], refs: References, task: str | None
) -> list[Doc]:
    for above, below in pairwise(rows):
        result = _pooled(compare(episodes, above["agent"], below["agent"], refs), task)
        above["vs_next"] = (
            None
            if result is None
            else {
                "agent": below["agent"],
                "pairs": result.pairs,
                "diff": result.diff,
                "diff_ci": list(result.diff_ci),
                "ahead": result.diff_ci[0] > 0,
            }
        )
    if rows:
        rows[-1]["vs_next"] = None
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _pooled(rows: list[Comparison], task: str | None) -> Comparison | None:
    want = task or "all"
    for r in rows:
        if r.task == want:
            return r
    return rows[0] if len(rows) == 1 and task is None else None


def _episode_row(e: Episode, refs: References) -> Doc:
    ref = refs.get(e.task)
    return {
        "agent": e.agent,
        "task": e.task,
        "seed": e.seed,
        "success": e.success,
        "truncated": e.truncated,
        "steps": e.steps,
        "return": e.total_return,
        "norm": ref.normalise(e.total_return) if ref else None,
        "invalid": e.invalid_actions,
        "tokens": e.input_tokens + e.output_tokens,
        "cost_usd": e.cost_usd,
        "path": episode_path(e.run_id, e.agent, e.task, e.seed),
    }


# ---------------------------------------------------------------- episode


def episode_doc(store: Store, run_id: str, agent: str, task: str, seed: int) -> Doc | None:
    ep = store.episode(run_id, agent, task, seed)
    if ep is None:
        return None
    ref = run_references(store, run_id, [task])[task]
    return {
        "run_id": run_id,
        "agent": ep.agent,
        "task": ep.task,
        "seed": ep.seed,
        "success": ep.success,
        "truncated": ep.truncated,
        "steps": ep.steps,
        "return": ep.total_return,
        "norm": ref.normalise(ep.total_return),
        "invalid": ep.invalid_actions,
        "input_tokens": ep.input_tokens,
        "output_tokens": ep.output_tokens,
        "cost_usd": ep.cost_usd,
        "duration_ms": ep.duration_ms,
        "trace": [
            {
                "t": s.t,
                "action": s.action,
                "thought": s.thought,
                "reward": s.reward,
                "invalid": s.invalid,
                "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens,
                "cost_usd": s.cost_usd,
                "latency_ms": s.latency_ms,
                "model_latency_ms": s.model_latency_ms,
                "observation": s.observation,
                "frame": parse_frame(s.observation),
            }
            for s in ep.trace
        ],
        "final": {"observation": ep.final_observation, "frame": _final_frame(ep)},
    }


def _final_frame(ep: Episode) -> Doc | None:
    """The last frame, with the hazard that ended the episode shown under the agent.

    The text map draws the agent over its cell, so a lava death would render
    as standing on floor. Terminal failures only happen on lava.
    """
    frame = parse_frame(ep.final_observation)
    if frame and frame["agent"] and not ep.success and not ep.truncated:
        x, y = frame["agent"]
        row = frame["grid"][y]
        frame["grid"][y] = row[:x] + "~" + row[x + 1 :]
        frame["died"] = True
    return frame


def parse_frame(observation: str) -> Doc | None:
    """Recover the grid from a gridworld observation's text map.

    The text is what the agent actually saw, so the picture drawn from it is
    exactly the agent's view. Returns ``None`` for observations without a map.
    ``grid`` rows hold one glyph per cell with the agent's glyph replaced by
    ``.`` - the agent is reported separately as ``agent`` and ``direction``.
    """
    lines = observation.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("Map ")) + 1
    except StopIteration:
        return None
    rows: list[list[str]] = []
    for line in lines[start:]:
        if line.startswith("Legend"):
            break
        rows.append(line.split(" "))
    agent, direction = None, None
    for y, row in enumerate(rows):
        for x, glyph in enumerate(row):
            if glyph in GLYPH_DIRECTIONS:
                agent, direction = [x, y], GLYPH_DIRECTIONS[glyph]
                row[x] = "."
    status = next((line for line in lines if line.startswith("You are at")), "")
    return {
        "grid": ["".join(row) for row in rows],
        "agent": agent,
        "direction": direction,
        "carrying": "Carrying key: yes" in status,
        "mission": lines[0].removeprefix("Mission: ").rstrip(".") if lines else "",
    }


# ----------------------------------------------------------------- export


def export(store: Store, out: Path, run_ids: list[str] | None = None) -> int:
    """Write every document for ``run_ids`` (default: all runs) under ``out``.

    Returns the number of files written. ``out`` becomes the site root: the
    built web app goes next to the ``data/`` directory written here.
    """
    index = runs_doc(store)
    if run_ids is not None:
        index["runs"] = [r for r in index["runs"] if r["id"] in run_ids]
    written = _write(out / "data" / "runs.json", index)
    for run in index["runs"]:
        board = leaderboard_doc(store, run["id"])
        if board is None:
            continue
        written += _write(out / run["leaderboard"], board)
        for row in board["episodes"]:
            doc = episode_doc(store, run["id"], row["agent"], row["task"], row["seed"])
            written += _write(out / row["path"], doc)
    return written


def _write(path: Path, doc: Doc | None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")))
    return 1
