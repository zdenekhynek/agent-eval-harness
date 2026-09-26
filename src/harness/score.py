"""Scoring. Success rate, return, steps and cost; bootstrapped CIs arrive in milestone 2."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from harness.trace import Episode


@dataclass(frozen=True)
class Score:
    agent: str
    task: str
    n: int
    success_rate: float
    mean_return: float
    mean_steps_on_success: float | None
    invalid_rate: float  # invalid actions per step taken
    mean_tokens: float | None  # input + output per episode, LLM agents only
    mean_cost_usd: float | None  # per episode, None when unpriced
    total_cost_usd: float | None


def score(episodes: list[Episode]) -> list[Score]:
    groups: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    for ep in episodes:
        groups[(ep.agent, ep.task)].append(ep)

    scores = []
    for (agent, task), eps in groups.items():
        wins = [e for e in eps if e.success]
        steps = sum(e.steps for e in eps)
        tokens = [e.input_tokens + e.output_tokens for e in eps]
        costs = [e.cost_usd for e in eps if e.cost_usd is not None]
        scores.append(
            Score(
                agent=agent,
                task=task,
                n=len(eps),
                success_rate=len(wins) / len(eps),
                mean_return=mean(e.total_return for e in eps),
                mean_steps_on_success=mean(e.steps for e in wins) if wins else None,
                invalid_rate=sum(e.invalid_actions for e in eps) / steps if steps else 0.0,
                mean_tokens=mean(tokens) if any(tokens) else None,
                mean_cost_usd=mean(costs) if costs else None,
                total_cost_usd=sum(costs) if costs else None,
            )
        )
    return sorted(scores, key=lambda s: (s.task, -s.success_rate, -s.mean_return, s.agent))


def format_table(scores: list[Score]) -> str:
    width = max([len("agent"), *(len(s.agent) for s in scores)])
    header = (
        f"{'task':<12} {'agent':<{width}} {'n':>4} {'success':>8} {'return':>7} "
        f"{'steps':>6} {'invalid':>8} {'tok/ep':>8} {'$/ep':>8}"
    )
    lines = [header, "-" * len(header)]
    for s in scores:
        steps = f"{s.mean_steps_on_success:.1f}" if s.mean_steps_on_success is not None else "-"
        tokens = f"{s.mean_tokens:,.0f}" if s.mean_tokens is not None else "-"
        cost = f"{s.mean_cost_usd:.4f}" if s.mean_cost_usd is not None else "-"
        lines.append(
            f"{s.task:<12} {s.agent:<{width}} {s.n:>4} {s.success_rate:>8.0%} "
            f"{s.mean_return:>7.3f} {steps:>6} {s.invalid_rate:>8.1%} {tokens:>8} {cost:>8}"
        )
    total = sum(s.total_cost_usd or 0.0 for s in scores)
    if total:
        lines.append(f"\ntotal cost ${total:.4f}")
    return "\n".join(lines)
