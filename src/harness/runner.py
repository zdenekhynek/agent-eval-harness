"""Runs the (agent x task x seed) matrix and records every episode.

Re-running with the same ``run_id`` is safe: episodes already in the store are
skipped, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from harness.agents import Agent, make_agent
from harness.envs import Env, InvalidActionError, make_env
from harness.store import Store
from harness.trace import Episode, StepRecord

# An agent that keeps emitting unparseable actions is stopped rather than
# burning the whole step budget (matters once each step costs tokens).
MAX_CONSECUTIVE_INVALID = 5


@dataclass
class RunSummary:
    run_id: str
    ran: int
    skipped: int


def run_episode(agent: Agent, env: Env, seed: int, run_id: str) -> Episode:
    ep = Episode(run_id=run_id, agent=agent.name, task=env.task_id, seed=seed)
    obs = env.reset(seed)
    agent.reset(env.task_id, seed)
    started = time.perf_counter()
    consecutive_invalid = 0

    for t in range(env.max_steps):
        t0 = time.perf_counter()
        decision = agent.act(obs)
        latency_ms = (time.perf_counter() - t0) * 1000
        record = StepRecord(
            t=t,
            observation=obs.text,
            action=decision.action,
            reward=0.0,
            thought=decision.thought,
            latency_ms=latency_ms,
        )
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
    on_episode: Callable[[Episode], None] | None = None,
) -> RunSummary:
    agents, tasks, seeds = list(agents), list(tasks), list(seeds)
    store.ensure_run(run_id, {"agents": agents, "tasks": tasks, "seeds": seeds})
    ran = skipped = 0
    for task in tasks:
        env = make_env(task)
        for agent_name in agents:
            agent = make_agent(agent_name)
            for seed in seeds:
                if store.has_episode(run_id, agent_name, task, seed):
                    skipped += 1
                    continue
                ep = run_episode(agent, env, seed, run_id)
                store.save_episode(ep)
                ran += 1
                if on_episode:
                    on_episode(ep)
    return RunSummary(run_id=run_id, ran=ran, skipped=skipped)
