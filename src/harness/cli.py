"""Command line: ``harness run | report | compare | show | list | export | serve``."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from harness.agents import AGENTS, Agent, canonical_name, is_llm, make_agent
from harness.agents.llm import DEFAULT_MODEL
from harness.envs import TASKS
from harness.llmapi import AnthropicClient, APIError, MissingAPIKeyError
from harness.runner import RunConfigError, run_matrix, run_references
from harness.score import compare, format_compare, format_overall, format_table, overall, score
from harness.store import Store
from harness.trace import Episode

DEFAULT_DB = "runs.db"
DEFAULT_WEB = Path(__file__).resolve().parents[2] / "web" / "dist"


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
        except RunConfigError as err:
            print(err, file=sys.stderr)
            return 2
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
        _print_report(store, run_id, store.episodes(run_id))
    return 0


def _print_report(store: Store, run_id: str, episodes: list[Episode]) -> None:
    refs = run_references(store, run_id, {e.task for e in episodes})
    print(format_table(score(episodes, refs)))
    table = format_overall(overall(episodes, refs))
    if table and len(refs) > 1:
        print()
        print(table)


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
        _print_report(store, run_id, episodes)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        run_id = _resolve_run(store, args.run_id)
        if run_id is None:
            return 1
        episodes = store.episodes(run_id)
        if args.tasks:
            episodes = [e for e in episodes if e.task in args.tasks]
        agents = {e.agent for e in episodes}
        missing = [a for a in (args.a, args.b) if a not in agents]
        if missing:
            known = ", ".join(sorted(agents)) or "none"
            print(f"not in run {run_id}: {', '.join(missing)} (agents: {known})", file=sys.stderr)
            return 1
        refs = run_references(store, run_id, {e.task for e in episodes})
        print(f"run {run_id}")
        print(format_compare(compare(episodes, args.a, args.b, refs), args.a, args.b))
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


def _cmd_export(args: argparse.Namespace) -> int:
    from harness.views import export

    with Store(args.db) as store:
        run_ids = args.run_ids or None
        if run_ids is None and args.latest:
            latest = _resolve_run(store, None)
            if latest is None:
                return 1
            run_ids = [latest]
        written = export(store, Path(args.out), run_ids)
    print(f"wrote {written} files to {args.out}/data")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from harness.server import create_app
    except ImportError:
        print("serving needs the extra: uv sync --extra serve", file=sys.stderr)
        return 2
    web = Path(args.web)
    if not (web / "index.html").exists():
        print(f"no built dashboard at {web}; serving the JSON API only", file=sys.stderr)
        print("build it with: npm --prefix web ci && npm --prefix web run build", file=sys.stderr)
        app = create_app(args.db)
    else:
        app = create_app(args.db, web)
    print(f"dashboard on http://{args.host}:{args.port}/  (db: {args.db})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
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

    cmp = sub.add_parser("compare", help="paired comparison of two agents on shared seeds")
    cmp.add_argument("a", help="agent as recorded, e.g. react:claude-sonnet-5")
    cmp.add_argument("b")
    cmp.add_argument("--tasks", type=_csv, help="limit to these tasks")
    cmp.add_argument("--run-id")
    cmp.set_defaults(func=_cmd_compare)

    show = sub.add_parser("show", help="step through one episode's trace")
    show.add_argument("agent", help="as recorded, e.g. scripted or react:claude-sonnet-5")
    show.add_argument("task")
    show.add_argument("seed", type=int)
    show.add_argument("--run-id")
    show.set_defaults(func=_cmd_show)

    exp = sub.add_parser("export", help="write the dashboard's JSON for static hosting")
    exp.add_argument("--out", default="site", help="site root; JSON lands in <out>/data")
    exp.add_argument("--run-ids", type=_csv, help="runs to export (default: all)")
    exp.add_argument("--latest", action="store_true", help="export only the latest run")
    exp.set_defaults(func=_cmd_export)

    srv = sub.add_parser("serve", help="serve the dashboard over the store, live")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--web", default=str(DEFAULT_WEB), help="built dashboard directory")
    srv.set_defaults(func=_cmd_serve)

    lst = sub.add_parser("list", help="list agents and tasks")
    lst.set_defaults(func=_cmd_list)
    return p


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Set ``KEY=value`` lines from ``path`` into the environment.

    Variables already set win, so a real environment always overrides the
    file. Keeping the API key here rather than in a shell profile keeps it
    out of every other program's environment. Returns the names it set.
    """
    file = Path(path)
    if not file.is_file():
        return []
    loaded = []
    for raw in file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
