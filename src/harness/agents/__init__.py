"""Agent registry.

Agents are named by spec: ``random``, ``scripted``, or ``<strategy>[:<model>]``
for LLM agents, e.g. ``react`` or ``react:claude-haiku-4-5``. An LLM agent's
recorded name always includes its model, so runs with different models sit
side by side on the leaderboard.
"""

from __future__ import annotations

from collections.abc import Callable

from harness.agents.base import Agent, Decision
from harness.agents.baselines import RandomAgent, ScriptedDoorKeyAgent
from harness.agents.llm import DEFAULT_MODEL, LLMAgent, ReActAgent, ZeroShotAgent
from harness.llmapi import AnthropicClient

BASELINES: dict[str, Callable[[], Agent]] = {
    "random": RandomAgent,
    "scripted": ScriptedDoorKeyAgent,
}
STRATEGIES: dict[str, type[LLMAgent]] = {
    "zero-shot": ZeroShotAgent,
    "react": ReActAgent,
}
AGENTS: tuple[str, ...] = (*BASELINES, *STRATEGIES)


def parse_spec(spec: str, default_model: str = DEFAULT_MODEL) -> tuple[str, str | None]:
    """Split ``react:claude-haiku-4-5`` into (strategy, model). Baselines have no model."""
    strategy, _, model = spec.partition(":")
    if strategy in BASELINES:
        if model:
            raise KeyError(f"baseline {strategy!r} takes no model")
        return strategy, None
    if strategy in STRATEGIES:
        return strategy, model or default_model
    raise KeyError(f"unknown agent {spec!r}; known: {', '.join(AGENTS)}")


def canonical_name(spec: str, default_model: str = DEFAULT_MODEL) -> str:
    strategy, model = parse_spec(spec, default_model)
    return strategy if model is None else f"{strategy}:{model}"


def is_llm(spec: str) -> bool:
    return spec.partition(":")[0] in STRATEGIES


def make_agent(
    spec: str,
    client: AnthropicClient | None = None,
    default_model: str = DEFAULT_MODEL,
    memory: int = 8,
) -> Agent:
    strategy, model = parse_spec(spec, default_model)
    if model is None:
        return BASELINES[strategy]()
    return STRATEGIES[strategy](client or AnthropicClient(), model=model, memory=memory)


__all__ = [
    "AGENTS",
    "BASELINES",
    "STRATEGIES",
    "Agent",
    "Decision",
    "canonical_name",
    "is_llm",
    "make_agent",
    "parse_spec",
]
