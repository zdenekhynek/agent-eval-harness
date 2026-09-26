"""Command line: ``harness run``, ``harness report``, ``harness show``, ``harness list``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime

from harness.agents import AGENTS, Agent, canonical_name, is_llm, make_agent
from harness.agents.llm import DEFAULT_MODEL
from harness.envs import TASKS
from harness.llmapi import AnthropicClient, APIError, MissingAPIKeyError
from harness.runner import run_matrix
from harness.score import format_table, score
from harness.store import Store
from harness.trace import Episode

DEFAULT_DB = "runs.db"


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        specs = [canonical_name(a, args.model) for a in args.agents]
    except KeyError as err:
        print(f"{err.args[0]} (see `harness list`)", file=sys.stderr)
        return 2
    unknown = [t for t in args.tasks if t not in TASKS]
    if unknown:
        print(f"unknown task: {', '.join(unknown)} (see `harness list`)", file=sys.stderr)
        return 2

    client: AnthropicClient | None = None
    if any(is_llm(s) for s in specs):
        try:
            client = AnthropicClient()
        except MissingAPIKeyError as err:
            print(err, file=sys.stderr)
            return 2

    def factory(spec: str) -> Agent:
        return make_agent(spec, client, memory=args.memory)

    def progress(ep: Episode) -> None:
        if not is_llm(ep.agent):
            return
        outcome = "solved" if ep.success else "failed"
        cost = f" ${ep.cost_usd:.4f}" if ep.cost_usd is not None else ""
        print(f"  {ep.agent} {ep.task} seed {ep.seed}: {outcome} in {ep.steps} steps{cost}")

    run_id = args.run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    with Store(args.db) as store:
        try:
            summary = run_matrix(
                store,
                run_id,
                specs,
                args.tasks,
                seeds,
                agent_factory=factory,
                max_steps=args.max_steps,
                max_cost_usd=args.max_cost,
                on_episode=progress,
            )
        except APIError as err:
            print(f"\n{err}", file=sys.stderr)
            print(f"finished episodes are saved; resume with --run-id {run_id}", file=sys.stderr)
            return 1
        print(f"{run_id}: ran {summary.ran} episodes, skipped {summary.skipped} already done")
        if summary.stopped_for_budget:
            print(
                f"stopped at the ${args.max_cost:.2f} cost cap; "
                f"raise --max-cost and resume with --run-id {run_id}"
            )
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
            if step.thought:
                print(f"--- thought ---\n{step.thought}")
            flag = "  [INVALID]" if step.invalid else ""
            usage = ""
            if step.input_tokens is not None:
                usage = f"  tokens={step.input_tokens}+{step.output_tokens}"
                if step.cost_usd is not None:
                    usage += f" ${step.cost_usd:.4f}"
            print(f">>> {step.action}{flag}  reward={step.reward:.3f}{usage}")
        print("--- final ---")
        print(ep.final_observation)
        outcome = "SUCCESS" if ep.success else ("TRUNCATED" if ep.truncated else "FAILED")
        cost = f", ${ep.cost_usd:.4f}" if ep.cost_usd is not None else ""
        print(f"{outcome} in {ep.steps} steps, return {ep.total_return:.3f}{cost}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    print("agents:", ", ".join(AGENTS), f"(LLM agents take :<model>, default {DEFAULT_MODEL})")
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
    run.add_argument(
        "--model", default=DEFAULT_MODEL, help="model for LLM agents given without :<model>"
    )
    run.add_argument("--max-steps", type=int, help="cap steps per episode (limits LLM cost)")
    run.add_argument("--max-cost", type=float, help="stop the run once it has spent this many USD")
    run.add_argument("--memory", type=int, default=8, help="recent steps shown to LLM agents")
    run.set_defaults(func=_cmd_run)

    report = sub.add_parser("report", help="score a run (latest by default)")
    report.add_argument("--run-id")
    report.set_defaults(func=_cmd_report)

    show = sub.add_parser("show", help="step through one episode's trace")
    show.add_argument("agent", help="as recorded, e.g. scripted or react:claude-sonnet-5")
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
