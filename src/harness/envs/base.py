"""Environment protocol shared by every task the harness can run.

An environment is deterministic given its seed: ``reset(seed)`` must always
produce the same layout, and the same action sequence must always produce the
same trajectory. That is what makes multi-seed comparisons between agents fair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Observation:
    """What an agent sees at one step.

    ``text`` is the rendering an LLM agent will be prompted with.
    ``state`` is structured, privileged state; only baseline agents (the
    scripted ceiling) may read it. LLM agents must work from ``text``.
    """

    text: str
    mission: str
    actions: tuple[str, ...]
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool  # reached a terminal state (goal reached)
    truncated: bool  # ran out of steps
    info: dict[str, Any] = field(default_factory=dict)


class InvalidActionError(ValueError):
    """Raised when an agent emits an action the environment does not accept."""


class Env(Protocol):
    task_id: str
    max_steps: int

    def reset(self, seed: int) -> Observation: ...

    def step(self, action: str) -> StepResult: ...
