"""Scoring: point estimates with 95% confidence intervals, normalised across tasks.

* **Success rate** gets a Wilson score interval - unlike a bootstrap it stays
  honest at 0/n and n/n, which is where small LLM runs usually land.
* **Normalised score** puts every task on one scale: 0 is the random agent's
  mean return, 1 the scripted solver's (``Reference``). Per (agent, task) it
  gets a percentile bootstrap over seeds; the cross-task ``overall`` score is
  the mean of per-task means with a bootstrap stratified by task, so a task
  with more seeds does not dominate.
* **Comparisons** between two agents are paired on seeds: both saw the same
  layouts, so the bootstrap runs on per-seed differences, which is far tighter
  than comparing two independent intervals.

Bootstraps use a fixed RNG seed, so a report is reproducible.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean

from harness.trace import Episode

Z95 = 1.959964
RESAMPLES = 2000
Interval = tuple[float, float]


@dataclass(frozen=True)
class Reference:
    """Mean returns that anchor a task's normalised score at 0 (floor) and 1 (ceiling)."""

    floor: float
    ceiling: float
    n: int

    def normalise(self, value: float) -> float | None:
        span = self.ceiling - self.floor
        return (value - self.floor) / span if span > 1e-9 else None


References = dict[str, Reference]


# ------------------------------------------------------------------ intervals


def wilson(successes: int, n: int, z: float = Z95) -> Interval:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return (lo, hi)


def bootstrap_ci(values: Sequence[float], resamples: int = RESAMPLES, seed: int = 0) -> Interval:
    """Percentile bootstrap CI for the mean."""
    return stratified_bootstrap_ci([values], resamples, seed)


def stratified_bootstrap_ci(
    groups: Sequence[Sequence[float]], resamples: int = RESAMPLES, seed: int = 0
) -> Interval:
    """Percentile bootstrap CI for the mean of group means, resampling within each group."""
    groups = [g for g in groups if g]
    if not groups:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    stats = sorted(mean(mean(rng.choices(g, k=len(g))) for g in groups) for _ in range(resamples))
    return (stats[int(0.025 * resamples)], stats[int(0.975 * resamples) - 1])


# --------------------------------------------------------------- per task


@dataclass(frozen=True)
class Score:
    agent: str
    task: str
    n: int
    success_rate: float
    success_ci: Interval
    mean_return: float
    norm: float | None  # normalised score, None without a reference for the task
    norm_ci: Interval | None
    mean_steps_on_success: float | None
    invalid_rate: float  # invalid actions per step taken
    mean_tokens: float | None  # input + output per episode, LLM agents only
    mean_cost_usd: float | None  # per episode, None when unpriced
    total_cost_usd: float | None


def _normalised(eps: list[Episode], ref: Reference | None) -> list[float] | None:
    if ref is None:
        return None
    values = [ref.normalise(e.total_return) for e in eps]
    return None if any(v is None for v in values) else [v for v in values if v is not None]


def score(episodes: list[Episode], references: References | None = None) -> list[Score]:
    references = references or {}
    groups: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    for ep in episodes:
        groups[(ep.agent, ep.task)].append(ep)

    scores = []
    for (agent, task), eps in groups.items():
        wins = [e for e in eps if e.success]
        steps = sum(e.steps for e in eps)
        tokens = [e.input_tokens + e.output_tokens for e in eps]
        costs = [e.cost_usd for e in eps if e.cost_usd is not None]
        norm = _normalised(eps, references.get(task))
        scores.append(
            Score(
                agent=agent,
                task=task,
                n=len(eps),
                success_rate=len(wins) / len(eps),
                success_ci=wilson(len(wins), len(eps)),
                mean_return=mean(e.total_return for e in eps),
                norm=mean(norm) if norm else None,
                norm_ci=bootstrap_ci(norm) if norm else None,
                mean_steps_on_success=mean(e.steps for e in wins) if wins else None,
                invalid_rate=sum(e.invalid_actions for e in eps) / steps if steps else 0.0,
                mean_tokens=mean(tokens) if any(tokens) else None,
                mean_cost_usd=mean(costs) if costs else None,
                total_cost_usd=sum(costs) if costs else None,
            )
        )
    return sorted(scores, key=lambda s: (s.task, -s.success_rate, -s.mean_return, s.agent))


# ---------------------------------------------------------------- overall


@dataclass(frozen=True)
class Overall:
    agent: str
    tasks: int
    n: int
    norm: float
    norm_ci: Interval


def overall(episodes: list[Episode], references: References) -> list[Overall]:
    """Mean normalised score across tasks, for agents that ran every task in ``episodes``.

    Tasks are weighted equally; an agent missing a task is left out rather
    than scored on an easier subset.
    """
    tasks = sorted({e.task for e in episodes})
    if not tasks or any(t not in references for t in tasks):
        return []
    by_agent: dict[str, dict[str, list[Episode]]] = defaultdict(lambda: defaultdict(list))
    for ep in episodes:
        by_agent[ep.agent][ep.task].append(ep)

    rows = []
    for agent, per_task in by_agent.items():
        if set(per_task) != set(tasks):
            continue
        groups = [_normalised(per_task[t], references[t]) or [] for t in tasks]
        rows.append(
            Overall(
                agent=agent,
                tasks=len(tasks),
                n=sum(len(g) for g in groups),
                norm=mean(mean(g) for g in groups),
                norm_ci=stratified_bootstrap_ci(groups),
            )
        )
    return sorted(rows, key=lambda r: (-r.norm, r.agent))


# ---------------------------------------------------------------- compare


@dataclass(frozen=True)
class Comparison:
    """``a`` minus ``b`` on the seeds both agents ran. ``task`` is ``"all"`` for the pooled row."""

    task: str
    pairs: int
    metric: str  # "norm" when references exist, else "success"
    a: float
    b: float
    diff: float
    diff_ci: Interval

    @property
    def verdict(self) -> str:
        lo, hi = self.diff_ci
        if lo > 0:
            return "A better"
        if hi < 0:
            return "B better"
        return "no clear difference"


def compare(
    episodes: list[Episode], a: str, b: str, references: References | None = None
) -> list[Comparison]:
    """Paired comparison of agents ``a`` and ``b``, per task and pooled across tasks."""
    references = references or {}
    by_key = {(e.agent, e.task, e.seed): e for e in episodes}
    tasks = sorted(
        {e.task for e in episodes if e.agent == a} & {e.task for e in episodes if e.agent == b}
    )
    metric = "norm" if tasks and all(t in references for t in tasks) else "success"

    def value(ep: Episode) -> float | None:
        if metric == "norm":
            return references[ep.task].normalise(ep.total_return)
        return float(ep.success)

    rows: list[Comparison] = []
    per_task: list[tuple[float, float, list[float]]] = []  # (mean a, mean b, diffs)
    for task in tasks:
        pairs = [
            (value(ep), value(by_key[(b, task, ep.seed)]))
            for ep in episodes
            if ep.agent == a and ep.task == task and (b, task, ep.seed) in by_key
        ]
        xs = [x for x, y in pairs if x is not None and y is not None]
        ys = [y for x, y in pairs if x is not None and y is not None]
        if not xs:
            continue
        diffs = [x - y for x, y in zip(xs, ys, strict=True)]
        rows.append(
            Comparison(task, len(xs), metric, mean(xs), mean(ys), mean(diffs), bootstrap_ci(diffs))
        )
        per_task.append((mean(xs), mean(ys), diffs))

    if len(per_task) > 1:
        groups = [diffs for _, _, diffs in per_task]
        rows.append(
            Comparison(
                "all",
                sum(len(g) for g in groups),
                metric,
                mean(x for x, _, _ in per_task),
                mean(y for _, y, _ in per_task),
                mean(mean(g) for g in groups),
                stratified_bootstrap_ci(groups),
            )
        )
    return rows


# ------------------------------------------------------------- formatting


def _ci(value: float, ci: Interval, pct: bool = False) -> str:
    if pct:
        return f"{value:.0%} [{ci[0]:.0%},{ci[1]:.0%}]"
    return f"{value:.2f} [{ci[0]:.2f},{ci[1]:.2f}]"


def format_table(scores: list[Score]) -> str:
    width = max([len("agent"), *(len(s.agent) for s in scores)])
    task_w = max([len("task"), *(len(s.task) for s in scores)])
    header = (
        f"{'task':<{task_w}} {'agent':<{width}} {'n':>4} {'success [95% CI]':>17} "
        f"{'norm [95% CI]':>18} {'return':>7} {'steps':>6} {'invalid':>8} "
        f"{'tok/ep':>8} {'$/ep':>8}"
    )
    lines = [header, "-" * len(header)]
    for s in scores:
        steps = f"{s.mean_steps_on_success:.1f}" if s.mean_steps_on_success is not None else "-"
        tokens = f"{s.mean_tokens:,.0f}" if s.mean_tokens is not None else "-"
        cost = f"{s.mean_cost_usd:.4f}" if s.mean_cost_usd is not None else "-"
        norm = _ci(s.norm, s.norm_ci) if s.norm is not None and s.norm_ci else "-"
        lines.append(
            f"{s.task:<{task_w}} {s.agent:<{width}} {s.n:>4} "
            f"{_ci(s.success_rate, s.success_ci, pct=True):>17} {norm:>18} "
            f"{s.mean_return:>7.3f} {steps:>6} {s.invalid_rate:>8.1%} {tokens:>8} {cost:>8}"
        )
    total = sum(s.total_cost_usd or 0.0 for s in scores)
    if total:
        lines.append(f"\ntotal cost ${total:.4f}")
    return "\n".join(lines)


def format_overall(rows: list[Overall]) -> str:
    if not rows:
        return ""
    width = max([len("agent"), *(len(r.agent) for r in rows)])
    header = f"{'agent':<{width}} {'tasks':>5} {'n':>5} {'norm [95% CI]':>18}"
    lines = [
        "overall: mean normalised score across tasks (0 = random, 1 = scripted)",
        header,
        "-" * len(header),
    ]
    lines += [
        f"{r.agent:<{width}} {r.tasks:>5} {r.n:>5} {_ci(r.norm, r.norm_ci):>18}" for r in rows
    ]
    return "\n".join(lines)


def format_compare(rows: list[Comparison], a: str, b: str) -> str:
    if not rows:
        return f"no seeds that both {a} and {b} ran"
    pct = rows[0].metric == "success"
    label = "success rate" if pct else "normalised score"
    task_w = max([len("task"), *(len(r.task) for r in rows)])
    spec = ".0%" if pct else ".2f"
    header = f"{'task':<{task_w}} {'pairs':>5} {'A':>6} {'B':>6} {'A - B [95% CI]':>22}  verdict"
    lines = [f"A = {a}\nB = {b}\npaired on seeds, metric: {label}", header, "-" * len(header)]
    for r in rows:
        diff = f"{r.diff:+.2f} [{r.diff_ci[0]:+.2f},{r.diff_ci[1]:+.2f}]"
        lines.append(
            f"{r.task:<{task_w}} {r.pairs:>5} {r.a:>6{spec}} {r.b:>6{spec}} {diff:>22}  {r.verdict}"
        )
    return "\n".join(lines)
