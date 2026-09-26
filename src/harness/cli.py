"""Command line: ``harness run``, ``harness report``, ``harness show``, ``harness list``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime

from harness.agents import AGENTS
from harness.envs import TASKS
from harness.runner import run_matrix
from harness.score import format_table, score
from harness.store import Store

DEFAULT_DB = "runs.db"


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _cmd_run(args: argparse.Namespace) -> int:
    unknown = [a for a in args.agents if a not in AGENTS]
    unknown += [t for t in args.tasks if t not in TASKS]
    if unknown:
        print(f"unknown agent/task: {', '.join(unknown)} (see `harness list`)", file=sys.stderr)
        return 2
    run_id = args.run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    with Store(args.db) as store:
        summary = run_matrix(store, run_id, args.agents, args.tasks, seeds)
        print(f"{run_id}: ran {summary.ran} episodes, skipped {summary.skipped} already done")
        print()
        print(format_table(score(store.episodes(run_id))))
    return 0


def _resolve_run(store: Store, run_id: str | None) -> str | None:
    resolved = run_id or store.latest_run_id()
    if resolved is None:
        print("no runs yet - start one with `harness run`", file=sys.stderr)
    return resolved


def _cmd_report(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        run_id = _resolve_run(store, args.run_id)
        if run_id is None:
            return 1
        episodes = store.episodes(run_id)
        if not episodes:
            print(f"run {run_id!r} has no episodes", file=sys.stderr)
            return 1
        print(f"run {run_id}")
        print(format_table(score(episodes)))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        run_id = _resolve_run(store, args.run_id)
        if run_id is None:
            return 1
        ep = store.episode(run_id, args.agent, args.task, args.seed)
        if ep is None:
            print("no such episode", file=sys.stderr)
            return 1
        for step in ep.trace:
            print(f"--- step {step.t} ---")
            print(step.observation)
            thought = f"  ({step.thought})" if step.thought else ""
            flag = "  [INVALID]" if step.invalid else ""
            print(f">>> {step.action}{thought}{flag}  reward={step.reward:.3f}")
        print("--- final ---")
        print(ep.final_observation)
        outcome = "SUCCESS" if ep.success else ("TRUNCATED" if ep.truncated else "FAILED")
        print(f"{outcome} in {ep.steps} steps, return {ep.total_return:.3f}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    print("agents:", ", ".join(sorted(AGENTS)))
    print("tasks: ", ", ".join(sorted(TASKS)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness", description="LLM-agent evaluation harness")
    p.add_argument("--db", default=DEFAULT_DB, help=f"SQLite file (default {DEFAULT_DB})")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run agents x tasks x seeds")
    run.add_argument("--agents", type=_csv, default=["random", "scripted"])
    run.add_argument("--tasks", type=_csv, default=["doorkey-5x5"])
    run.add_argument("--seeds", type=int, default=20, help="number of seeds")
    run.add_argument("--seed-start", type=int, default=0)
    run.add_argument("--run-id", help="reuse an id to resume an interrupted run")
    run.set_defaults(func=_cmd_run)

    report = sub.add_parser("report", help="score a run (latest by default)")
    report.add_argument("--run-id")
    report.set_defaults(func=_cmd_report)

    show = sub.add_parser("show", help="step through one episode's trace")
    show.add_argument("agent")
    show.add_argument("task")
    show.add_argument("seed", type=int)
    show.add_argument("--run-id")
    show.set_defaults(func=_cmd_show)

    lst = sub.add_parser("list", help="list agents and tasks")
    lst.set_defaults(func=_cmd_list)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
