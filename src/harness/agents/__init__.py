"""Agent registry. LLM prompting strategies will register here alongside baselines."""

from __future__ import annotations

from collections.abc import Callable

from harness.agents.base import Agent, Decision
from harness.agents.baselines import RandomAgent, ScriptedDoorKeyAgent

AGENTS: dict[str, Callable[[], Agent]] = {
    "random": RandomAgent,
    "scripted": ScriptedDoorKeyAgent,
}


def make_agent(name: str) -> Agent:
    try:
        return AGENTS[name]()
    except KeyError:
        raise KeyError(f"unknown agent {name!r}; known: {sorted(AGENTS)}") from None


__all__ = ["AGENTS", "Agent", "Decision", "make_agent"]
