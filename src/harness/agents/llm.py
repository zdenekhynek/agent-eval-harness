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
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
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
class _Call:
    action: str
    thought: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


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
        started = time.perf_counter()
        action, thought, calls = self._decide(observation)
        self._history.append(_Memory(self._t, action, thought))
        self._t += 1
        costs = [c.cost_usd for c in calls if c.cost_usd is not None]
        return Decision(
            action=action,
            thought=thought,
            input_tokens=sum(c.input_tokens for c in calls),
            output_tokens=sum(c.output_tokens for c in calls),
            cost_usd=sum(costs) if costs else None,
            # Wall time across the decision's calls, which may run in parallel.
            model_latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _decide(self, observation: Observation) -> tuple[str, str | None, list[_Call]]:
        """Choose an action. Strategies override this; most make one ``_call``."""
        call = self._call(observation)
        return call.action, call.thought, [call]

    def _call(
        self, observation: Observation, extra: str = "", act: bool = True, max_tokens: int = 0
    ) -> _Call:
        """One model request. ``extra`` is appended to the user prompt; with
        ``act=False`` the model answers in text only (no tool)."""
        user = self.user_prompt(observation) + (f"\n\n{extra}" if extra else "")
        tool_choice: dict[str, Any] | None = None
        if act:
            forced = self.force_tool and not self.thinking
            tool_choice = {"type": "tool", "name": "act"} if forced else {"type": "auto"}
        response = self.client.messages(
            model=self.model,
            system=self.system_prompt(observation),
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens or (THINKING_MAX_TOKENS if self.thinking else self.max_tokens),
            tools=[self.tool(observation)] if act else None,
            tool_choice=tool_choice,
            temperature=self.temperature,
            thinking={"type": "adaptive"} if self.thinking else {"type": "disabled"},
        )
        action = NO_ACTION
        for block in response.tool_uses():
            if block.get("name") == "act":
                action = str((block.get("input") or {}).get("action", NO_ACTION))
                break
        return _Call(
            action=action,
            thought=response.text() or None,
            input_tokens=response.usage.input_tokens + response.usage.cache_read_input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost_usd(self.model, response.usage),
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


# --------------------------------------------------------------- few-shot

EXAMPLE_TASKS = ("doorkey-5x5", "lavagap-5x5")
EXAMPLE_SEEDS = range(900, 950)  # never evaluated: runs use 0-99 (dev) and 1000+ (held out)
DIRECTION_NAMES = ("east", "south", "west", "north")


@cache
def few_shot_examples() -> str:
    """Worked examples drawn from scripted episodes on seeds no run evaluates.

    One of each lesson an LLM agent tends to get wrong: which way ``left`` and
    ``right`` turn, picking up the key, unlocking the door, and refusing to
    step into lava. Generated, not hand-written, so they are always correct for
    the current environments.
    """
    from harness.agents.baselines import ScriptedAgent
    from harness.envs import make_env

    wanted = ["turn", "pickup", "toggle", "lava"]
    examples: dict[str, str] = {}
    episodes = ((task, seed) for task in EXAMPLE_TASKS for seed in EXAMPLE_SEEDS)
    for task, seed in episodes:
        if len(examples) == len(wanted):
            break
        env, agent = make_env(task), ScriptedAgent()
        obs = env.reset(seed)
        for _ in range(env.max_steps):
            decision = agent.act(obs)
            kind = _example_kind(obs, decision.action)
            if kind in wanted and kind not in examples:
                examples[kind] = (
                    f"Observation:\n{obs.text}\nCorrect action: {decision.action}\n"
                    f"Why: {_why(obs, decision.action, decision.thought or '')}"
                )
            result = env.step(decision.action)
            if result.terminated or result.truncated:
                break
            obs = result.observation
    body = "\n\n".join(
        f"Example {i}\n{examples[k]}" for i, k in enumerate(wanted, start=1) if k in examples
    )
    return "Here are worked examples from other layouts:\n\n" + body


def _ahead_glyph(obs: Observation) -> str:
    s = obs.state
    (x, y), d = s["agent"], s["direction"]
    dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[d]
    return str(s["grid"][y + dy][x + dx])


def _example_kind(obs: Observation, action: str) -> str:
    if _ahead_glyph(obs) == "~" and action in ("left", "right"):
        return "lava"
    if action in ("left", "right"):
        return "turn"
    return action


def _why(obs: Observation, action: str, goal: str) -> str:
    d = int(obs.state["direction"])
    if _ahead_glyph(obs) == "~":
        turned = DIRECTION_NAMES[(d + (1 if action == "right" else -1)) % 4]
        return (
            f"Lava is directly ahead; stepping forward would end the episode. "
            f"'{action}' turns you from {DIRECTION_NAMES[d]} to {turned}, away from it."
        )
    if action in ("left", "right"):
        turned = DIRECTION_NAMES[(d + (1 if action == "right" else -1)) % 4]
        way = "clockwise" if action == "right" else "counter-clockwise"
        return (
            f"You face {DIRECTION_NAMES[d]} and need to {goal}. '{action}' turns {way}, "
            f"from {DIRECTION_NAMES[d]} to {turned}."
        )
    if action == "pickup":
        return "The key is directly ahead, so pick it up before going to the door."
    if action == "toggle":
        return "You carry the key and the locked door is directly ahead, so toggle unlocks it."
    return f"The cell ahead is free and on the way to {goal or 'the goal'}."


class FewShotAgent(ZeroShotAgent):
    """Zero-shot plus worked examples: isolates what in-context examples add."""

    strategy = "few-shot"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.system = SYSTEM_BASE + "\n\n" + few_shot_examples()


# -------------------------------------------------------------- reflexion

REFLECT_PROMPT = """You have been here before: this exact situation (same map, position, facing \
and inventory) already occurred at step {then}, so your recent actions went in a circle.
Do not choose an action yet. In two or three sentences, say what went wrong and what you will \
do differently from now on."""


class ReflexionAgent(ReActAgent):
    """ReAct with in-episode self-reflection.

    When the agent revisits a situation it has already been in (a loop, judged
    from the observation text alone), it first writes a short lesson about what
    went wrong; lessons stay in its prompt for the rest of the episode. This is
    Reflexion's self-critique inside one episode rather than across retries,
    so every strategy still gets exactly one attempt per seed.
    """

    strategy = "reflexion"
    max_lessons = 3
    cooldown = 4  # steps between reflections

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._seen: dict[str, int] = {}
        self._lessons: list[str] = []
        self._last_reflection = -self.cooldown

    def reset(self, task_id: str, seed: int) -> None:
        super().reset(task_id, seed)
        self._seen.clear()
        self._lessons.clear()
        self._last_reflection = -self.cooldown

    def user_prompt(self, observation: Observation) -> str:
        prompt = super().user_prompt(observation)
        if not self._lessons:
            return prompt
        lessons = "\n".join(f"- {lesson}" for lesson in self._lessons)
        return f"Lessons you wrote earlier in this episode:\n{lessons}\n\n{prompt}"

    def _decide(self, observation: Observation) -> tuple[str, str | None, list[_Call]]:
        calls: list[_Call] = []
        key = _situation(observation.text)
        then = self._seen.get(key)
        self._seen[key] = self._t
        reflection = None
        if then is not None and self._t - self._last_reflection >= self.cooldown:
            call = self._call(
                observation, REFLECT_PROMPT.format(then=then), act=False, max_tokens=300
            )
            calls.append(call)
            reflection = (call.thought or "").strip()
            if reflection:
                self._lessons = [*self._lessons, _clip(reflection, 300)][-self.max_lessons :]
            self._last_reflection = self._t
        act = self._call(observation)
        calls.append(act)
        thought = act.thought
        if reflection:
            thought = f"Reflection: {reflection}\n\n{thought or ''}".strip()
        return act.action, thought, calls


def _situation(text: str) -> str:
    """The observation without its step counter: equal means the same situation."""
    return "\n".join(line for line in text.splitlines() if not line.startswith("Step "))


# ------------------------------------------------------- self-consistency


class SelfConsistencyAgent(ReActAgent):
    """ReAct sampled several times in parallel; the majority action wins.

    Ties go to the action sampled first. The recorded thought shows the vote
    and the reasoning of the first sample that chose the winning action.
    """

    strategy = "self-consistency"
    samples = 3

    def _decide(self, observation: Observation) -> tuple[str, str | None, list[_Call]]:
        with ThreadPoolExecutor(self.samples) as pool:
            calls = list(pool.map(lambda _: self._call(observation), range(self.samples)))
        votes = Counter(c.action for c in calls)
        top = max(votes.values())
        winner = next(c for c in calls if votes[c.action] == top)
        tally = ", ".join(f"{a} x{n}" for a, n in votes.most_common())
        thought = f"Votes: {tally}.\n\n{winner.thought or ''}".strip()
        return winner.action, thought, calls
