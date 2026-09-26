"""Scoring. Milestone 0 has success rate; bootstrapped CIs arrive in milestone 2."""

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


def score(episodes: list[Episode]) -> list[Score]:
    groups: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    for ep in episodes:
        groups[(ep.agent, ep.task)].append(ep)

    scores = []
    for (agent, task), eps in groups.items():
        wins = [e for e in eps if e.success]
        scores.append(
            Score(
                agent=agent,
                task=task,
                n=len(eps),
                success_rate=len(wins) / len(eps),
                mean_return=mean(e.total_return for e in eps),
                mean_steps_on_success=mean(e.steps for e in wins) if wins else None,
            )
        )
    return sorted(scores, key=lambda s: (s.task, -s.success_rate, -s.mean_return, s.agent))


def format_table(scores: list[Score]) -> str:
    header = f"{'task':<14} {'agent':<12} {'n':>4} {'success':>8} {'return':>7} {'steps':>6}"
    lines = [header, "-" * len(header)]
    for s in scores:
        steps = f"{s.mean_steps_on_success:.1f}" if s.mean_steps_on_success is not None else "-"
        lines.append(
            f"{s.task:<14} {s.agent:<12} {s.n:>4} {s.success_rate:>8.0%} "
            f"{s.mean_return:>7.3f} {steps:>6}"
        )
    return "\n".join(lines)
