"""Runs the (agent x task x seed) matrix and records every episode.

Re-running with the same ``run_id`` is safe: episodes already in the store are
skipped, so an interrupted run (or one stopped by the cost cap, or by an API
outage) resumes where it stopped.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from harness.agents import Agent, canonical_name, make_agent
from harness.envs import Env, InvalidActionError, make_env
from harness.store import Store
from harness.trace import Episode, StepRecord

# An agent that keeps emitting unusable actions is stopped rather than
# burning the whole step budget (and, for LLM agents, tokens).
MAX_CONSECUTIVE_INVALID = 5


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
            ep.success = result.terminated
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
    store.ensure_run(
        run_id,
        {"agents": list(names.values()), "tasks": tasks, "seeds": seeds, "max_steps": max_steps},
    )
    summary = RunSummary(run_id=run_id, ran=0, skipped=0)
    built: dict[str, Agent] = {}

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
                agent = built.get(spec) or built.setdefault(spec, agent_factory(spec))
                ep = run_episode(agent, env, seed, run_id, max_steps=max_steps)
                store.save_episode(ep)
                summary.ran += 1
                summary.cost_usd += ep.cost_usd or 0.0
                if on_episode:
                    on_episode(ep)
    return summary
