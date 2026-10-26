"""LLM agents. Each prompting strategy is its own agent so the leaderboard compares them.

Every step is a fresh, stateless request: system prompt + a short history of
recent moves + the current observation. That keeps per-step cost flat however
long an episode runs, and makes each decision reproducible from its trace.

Strategies:

* ``zero-shot`` - the model must call the ``act`` tool straight away (forced
  tool choice), so there is no visible reasoning. Measures raw capability.
* ``react`` - the model reasons in plain text first, then calls ``act``.
  The reasoning is recorded as the step's thought.

Extended thinking is a separate, explicit variable. By default every strategy
sends ``thinking: disabled``: some models (Sonnet 5) otherwise think
privately by default, which would hand ReAct a hidden reasoning channel that
forced-tool zero-shot never gets and confound the comparison. A ``+think``
suffix (``react+think``) turns on adaptive thinking as its own leaderboard
entry; the API does not allow forced tool choice alongside thinking, so
``zero-shot+think`` calls ``act`` with ``tool_choice: auto``.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from typing import Any

from harness.agents.base import Decision
from harness.envs.base import Observation
from harness.llmapi import AnthropicClient, cost_usd

DEFAULT_MODEL = "claude-sonnet-5"
NO_ACTION = "<no action>"
# Bump when the user-prompt layout in ``LLMAgent.user_prompt`` changes; the
# prompt texts themselves are hashed directly into each agent's fingerprint.
PROMPT_VERSION = 2

ACTION_HELP = {
    "left": "turn 90 degrees counter-clockwise on the map (east -> north -> west -> south)",
    "right": "turn 90 degrees clockwise on the map (east -> south -> west -> north)",
    "forward": "move one cell in the direction you face, unless the cell is blocked",
    "pickup": "pick up the object directly ahead of you",
    "toggle": "open or close the door directly ahead of you; a locked door needs the key",
}

SYSTEM_BASE = """You are an agent acting in a small gridworld, one action per turn.

Each turn you get the current observation and a list of your most recent actions.
Rules:
- You can only interact with the cell directly ahead of you, so turn to face things first.
- Walls, objects lying on the floor (such as keys) and closed or locked doors block movement.
- Read the legend: some cells, such as lava, end the episode in failure if you step onto them.
- If the map, your position, your facing and what you carry did not change after an
  action, that action had no effect - try something else.

Choose your action by calling the `act` tool exactly once."""

REACT_SUFFIX = """

Before calling `act`, think briefly in plain text: where you are, what you need next
(for example a key, a door or the goal), and which single action gets you closer.
Keep it to a few sentences."""


@dataclass
class _Memory:
    t: int
    action: str
    thought: str | None


THINKING_SUFFIX = "+think"
THINKING_MAX_TOKENS = 4096  # thinking tokens count toward max_tokens


class LLMAgent:
    strategy = "llm"
    system = SYSTEM_BASE
    force_tool = True
    max_tokens = 256

    def __init__(
        self,
        client: AnthropicClient,
        model: str = DEFAULT_MODEL,
        memory: int = 8,
        temperature: float | None = None,
        thinking: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.memory_size = memory
        self.temperature = temperature
        self.thinking = thinking
        suffix = THINKING_SUFFIX if thinking else ""
        self.name = f"{self.strategy}{suffix}:{model}"
        self._history: deque[_Memory] = deque(maxlen=memory)
        self._t = 0

    def reset(self, task_id: str, seed: int) -> None:
        self._history.clear()
        self._t = 0

    # ---------------------------------------------------------------- prompts

    def fingerprint(self) -> str:
        """Short hash of the prompts and settings; a run refuses to resume if it changes."""
        spec = {
            "prompt_version": PROMPT_VERSION,
            "strategy": self.strategy,
            "model": self.model,
            "system": self.system,
            "action_help": ACTION_HELP,
            "force_tool": self.force_tool,
            "max_tokens": self.max_tokens,
            "memory": self.memory_size,
            "temperature": self.temperature,
            "thinking": self.thinking,
        }
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]

    def system_prompt(self, observation: Observation) -> str:
        return self.system

    def user_prompt(self, observation: Observation) -> str:
        parts = []
        if self._history:
            lines = []
            for m in self._history:
                line = f"- step {m.t}: {m.action}"
                if m.thought:
                    line += f"  (you thought: {_clip(m.thought, 160)})"
                lines.append(line)
            parts.append("Your recent actions, oldest first:\n" + "\n".join(lines))
        else:
            parts.append("This is the first step of the episode.")
        parts.append("Current observation:\n" + observation.text)
        return "\n\n".join(parts)

    def tool(self, observation: Observation) -> dict[str, Any]:
        described = "\n".join(f"- {a}: {ACTION_HELP.get(a, a)}" for a in observation.actions)
        return {
            "name": "act",
            "description": "Take one action in the environment.\n" + described,
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(observation.actions)},
                },
                "required": ["action"],
            },
        }

    # -------------------------------------------------------------------- act

    def act(self, observation: Observation) -> Decision:
        response = self.client.messages(
            model=self.model,
            system=self.system_prompt(observation),
            messages=[{"role": "user", "content": self.user_prompt(observation)}],
            max_tokens=THINKING_MAX_TOKENS if self.thinking else self.max_tokens,
            tools=[self.tool(observation)],
            tool_choice=(
                {"type": "tool", "name": "act"}
                if self.force_tool and not self.thinking
                else {"type": "auto"}
            ),
            temperature=self.temperature,
            thinking={"type": "adaptive"} if self.thinking else {"type": "disabled"},
        )
        action = NO_ACTION
        for block in response.tool_uses():
            if block.get("name") == "act":
                action = str((block.get("input") or {}).get("action", NO_ACTION))
                break
        thought = response.text() or None

        self._history.append(_Memory(self._t, action, thought))
        self._t += 1
        return Decision(
            action=action,
            thought=thought,
            input_tokens=response.usage.input_tokens + response.usage.cache_read_input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost_usd(self.model, response.usage),
            model_latency_ms=response.latency_ms,
        )


class ZeroShotAgent(LLMAgent):
    strategy = "zero-shot"
    force_tool = True
    max_tokens = 256


class ReActAgent(LLMAgent):
    strategy = "react"
    system = SYSTEM_BASE + REACT_SUFFIX
    force_tool = False
    max_tokens = 1024


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
