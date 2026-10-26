"""Runs the (agent x task x seed) matrix and records every episode.

Re-running with the same ``run_id`` is safe: episodes already in the store are
skipped, so an interrupted run (or one stopped by the cost cap, or by an API
outage) resumes where it stopped. A resume is refused if an agent's prompt
fingerprint or the step cap changed, so one run never mixes two experiments.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from statistics import mean
from typing import Any

from harness import __version__
from harness.agents import Agent, canonical_name, make_agent
from harness.envs import Env, InvalidActionError, make_env
from harness.score import Reference, References
from harness.store import Store
from harness.trace import Episode, StepRecord

# An agent that keeps emitting unusable actions is stopped rather than
# burning the whole step budget (and, for LLM agents, tokens).
MAX_CONSECUTIVE_INVALID = 5

# Normalisation anchors come from a fixed seed set, never from the run being
# scored, so a task's 0 and 1 mean the same thing in every report.
REFERENCE_SEEDS = range(100)


class RunConfigError(ValueError):
    """Resuming a run with settings that would make its episodes incomparable."""


@dataclass
class RunSummary:
    run_id: str
    ran: int
    skipped: int
    cost_usd: float = 0.0
    stopped_for_budget: bool = False


def run_episode(
    agent: Agent, env: Env, seed: int, run_id: str, max_steps: int | None = None
) -> Episode:
    ep = Episode(run_id=run_id, agent=agent.name, task=env.task_id, seed=seed)
    obs = env.reset(seed)
    agent.reset(env.task_id, seed)
    limit = env.max_steps if max_steps is None else min(max_steps, env.max_steps)
    started = time.perf_counter()
    consecutive_invalid = 0

    for t in range(limit):
        t0 = time.perf_counter()
        decision = agent.act(obs)
        record = StepRecord(
            t=t,
            observation=obs.text,
            action=decision.action,
            reward=0.0,
            thought=decision.thought,
            latency_ms=(time.perf_counter() - t0) * 1000,
            input_tokens=decision.input_tokens,
            output_tokens=decision.output_tokens,
            cost_usd=decision.cost_usd,
            model_latency_ms=decision.model_latency_ms,
        )
        ep.add_usage(record)
        try:
            result = env.step(decision.action)
        except InvalidActionError:
            record.invalid = True
            ep.invalid_actions += 1
            ep.trace.append(record)
            consecutive_invalid += 1
            if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                ep.truncated = True
                break
            continue

        consecutive_invalid = 0
        record.reward = result.reward
        ep.trace.append(record)
        ep.total_return += result.reward
        obs = result.observation
        if result.terminated or result.truncated:
            ep.success = result.success
            ep.truncated = result.truncated
            break
    else:
        ep.truncated = True

    ep.steps = len(ep.trace)
    ep.final_observation = obs.text
    ep.duration_ms = (time.perf_counter() - started) * 1000
    return ep


def run_matrix(
    store: Store,
    run_id: str,
    agents: Iterable[str],
    tasks: Iterable[str],
    seeds: Iterable[int],
    agent_factory: Callable[[str], Agent] = make_agent,
    max_steps: int | None = None,
    max_cost_usd: float | None = None,
    on_episode: Callable[[Episode], None] | None = None,
) -> RunSummary:
    """Run every (agent, task, seed) combination not already in the store.

    ``max_cost_usd`` is checked between episodes: the episode that crosses the
    cap finishes, then the run stops. Resume with the same ``run_id``.
    """
    specs, tasks, seeds = list(agents), list(tasks), list(seeds)
    names = {spec: canonical_name(spec) for spec in specs}
    built = {spec: agent_factory(spec) for spec in specs}
    _record_config(store, run_id, built, names, tasks, seeds, max_steps)
    summary = RunSummary(run_id=run_id, ran=0, skipped=0)

    for task in tasks:
        env = make_env(task)
        for spec in specs:
            for seed in seeds:
                if store.has_episode(run_id, names[spec], task, seed):
                    summary.skipped += 1
                    continue
                if max_cost_usd is not None and summary.cost_usd >= max_cost_usd:
                    summary.stopped_for_budget = True
                    return summary
                ep = run_episode(built[spec], env, seed, run_id, max_steps=max_steps)
                store.save_episode(ep)
                summary.ran += 1
                summary.cost_usd += ep.cost_usd or 0.0
                if on_episode:
                    on_episode(ep)
    return summary


def fingerprint(agent: Agent) -> str | None:
    """Hash of everything that shapes an agent's behaviour, if it exposes one."""
    fp = getattr(agent, "fingerprint", None)
    return str(fp()) if callable(fp) else None


def _record_config(
    store: Store,
    run_id: str,
    agents: dict[str, Agent],
    names: dict[str, str],
    tasks: list[str],
    seeds: list[int],
    max_steps: int | None,
) -> None:
    fingerprints = {names[spec]: fingerprint(a) for spec, a in agents.items()}
    old = store.run_config(run_id)
    if old is None:
        store.ensure_run(
            run_id,
            {
                "harness_version": __version__,
                "agents": list(names.values()),
                "tasks": tasks,
                "seeds": seeds,
                "max_steps": max_steps,
                "fingerprints": fingerprints,
            },
        )
        return

    if old.get("max_steps") != max_steps:

        def cap(v: int | None) -> str:
            return "no --max-steps" if v is None else f"--max-steps {v}"

        raise RunConfigError(
            f"run {run_id!r} used {cap(old.get('max_steps'))}, not {cap(max_steps)}; "
            "resume with the same cap or start a new run"
        )
    old_fps: dict[str, Any] = old.get("fingerprints", {})
    changed = [a for a, fp in fingerprints.items() if a in old_fps and old_fps[a] != fp]
    if changed:
        raise RunConfigError(
            f"prompt or settings changed since run {run_id!r} started for: "
            f"{', '.join(changed)}; start a new run so results are not mixed"
        )
    old["agents"] = list(dict.fromkeys([*old.get("agents", []), *names.values()]))
    old["tasks"] = list(dict.fromkeys([*old.get("tasks", []), *tasks]))
    old["seeds"] = sorted({*old.get("seeds", []), *seeds})
    old["fingerprints"] = {**old_fps, **fingerprints}
    store.update_run_config(run_id, old)


def compute_reference(
    task: str, max_steps: int | None = None, seeds: Iterable[int] = REFERENCE_SEEDS
) -> Reference:
    """Mean return of the random floor and scripted ceiling on ``task``.

    ``max_steps`` must match the run being scored: a step cap truncates random
    episodes that would otherwise have stumbled onto the goal, which lowers
    the floor. Anchors computed without the cap would misplace 0.
    """
    env = make_env(task)
    seeds = list(seeds)

    def mean_return(agent: str) -> float:
        return mean(
            run_episode(make_agent(agent), env, s, "ref", max_steps=max_steps).total_return
            for s in seeds
        )

    return Reference(floor=mean_return("random"), ceiling=mean_return("scripted"), n=len(seeds))


def references(store: Store, tasks: Iterable[str], max_steps: int | None = None) -> References:
    """Normalisation anchors for ``tasks`` under a step cap, cached per harness version."""
    key = __version__ if max_steps is None else f"{__version__}+cap{max_steps}"
    refs: References = {}
    for task in sorted(set(tasks)):
        ref = store.reference(task, key)
        if ref is None:
            ref = compute_reference(task, max_steps)
            store.save_reference(task, key, ref)
        refs[task] = ref
    return refs


def run_references(store: Store, run_id: str, tasks: Iterable[str]) -> References:
    """Anchors for a run's episodes, under the same ``--max-steps`` the run used."""
    config = store.run_config(run_id) or {}
    return references(store, tasks, config.get("max_steps"))
