"""Agent protocol. Every agent - scripted baseline or LLM strategy - fits this."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from harness.envs.base import Observation


@dataclass(frozen=True)
class Decision:
    action: str
    thought: str | None = None  # an LLM agent's reasoning, shown in the trace viewer


class Agent(Protocol):
    name: str

    def reset(self, task_id: str, seed: int) -> None: ...

    def act(self, observation: Observation) -> Decision: ...
