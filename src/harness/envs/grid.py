"""Shared MiniGrid mechanics for every gridworld task.

Turn-then-move movement, one-cell-ahead interaction, MiniGrid's success reward
``1 - 0.9 * steps / max_steps``, and a text rendering an LLM agent can read.
Subclasses lay out the grid in ``_generate`` and describe cells in ``cell``;
anything beyond turning and moving (keys, doors) goes in ``_interact``.

Cells are glyphs. The glyph under a position decides everything generic:
``WALKABLE`` glyphs can be entered, ``G`` ends the episode in success, ``~``
(lava) ends it in failure. The observation's privileged ``state["grid"]`` is
the glyph map without the agent, which is all the scripted solver needs.
"""

from __future__ import annotations

import random
from typing import Any

from harness.envs.base import InvalidActionError, Observation, StepResult

# Direction index -> (dx, dy). 0 = east, 1 = south, 2 = west, 3 = north,
# matching MiniGrid's convention (y grows downwards).
DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))
DIRECTION_NAMES: tuple[str, ...] = ("east", "south", "west", "north")
AGENT_GLYPHS: tuple[str, ...] = (">", "v", "<", "^")

WALKABLE = frozenset({".", "G", "_", "~"})  # lava is walkable - once
GOAL, LAVA = "G", "~"

Pos = tuple[int, int]


class GridEnv:
    """A ``size`` x ``size`` room (outer walls included) with a goal square."""

    name = "grid"
    mission = "get to the goal"
    legend = "# wall, . floor, G goal"
    actions: tuple[str, ...] = ("left", "right", "forward")
    steps_per_cell = 4  # MiniGrid's max_steps = steps_per_cell * size * size

    def __init__(self, size: int) -> None:
        if size < 5:
            raise ValueError(f"{self.name} needs size >= 5")
        self.size = size
        self.task_id = f"{self.name}-{size}x{size}"
        self.max_steps = self.steps_per_cell * size * size
        self.agent: Pos = (1, 1)
        self.direction = 0
        self.goal: Pos = (size - 2, size - 2)
        self._steps = 0
        self._ready = False

    # ------------------------------------------------------------------ hooks

    def _generate(self, rng: random.Random) -> None:
        """Lay out the grid and place the agent. Must only draw from ``rng``."""
        raise NotImplementedError

    def cell(self, pos: Pos) -> str:
        """Glyph for ``pos``, ignoring the agent."""
        x, y = pos
        n = self.size
        if x <= 0 or y <= 0 or x >= n - 1 or y >= n - 1:
            return "#"
        return GOAL if pos == self.goal else "."

    def _interact(self, action: str, ahead: Pos) -> None:
        """Handle a non-movement action on the cell ahead."""

    def _status(self) -> str:
        """Extra sentence(s) for the observation, after what is ahead."""
        return ""

    def _extra_state(self) -> dict[str, Any]:
        return {}

    # ------------------------------------------------------------- lifecycle

    def reset(self, seed: int) -> Observation:
        self._generate(random.Random(f"{self.task_id}:{seed}"))
        self._steps = 0
        self._ready = True
        return self._observe()

    def step(self, action: str) -> StepResult:
        self._require_reset()
        if action not in self.actions:
            raise InvalidActionError(f"unknown action {action!r}; expected one of {self.actions}")

        self._steps += 1
        ahead = self.ahead()
        if action == "left":
            self.direction = (self.direction - 1) % 4
        elif action == "right":
            self.direction = (self.direction + 1) % 4
        elif action == "forward":
            if self.passable(ahead):
                self.agent = ahead
        else:
            self._interact(action, ahead)

        here = self.cell(self.agent)
        success = here == GOAL
        terminated = success or here == LAVA
        reward = 1.0 - 0.9 * (self._steps / self.max_steps) if success else 0.0
        truncated = not terminated and self._steps >= self.max_steps
        return StepResult(self._observe(), reward, terminated, truncated, success=success)

    # ---------------------------------------------------------------- queries

    def passable(self, pos: Pos) -> bool:
        """Can the agent move onto ``pos`` in the current state?"""
        return self.cell(pos) in WALKABLE

    def ahead(self) -> Pos:
        dx, dy = DIRECTIONS[self.direction]
        return (self.agent[0] + dx, self.agent[1] + dy)

    def grid(self) -> list[str]:
        """Glyph rows without the agent."""
        return ["".join(self.cell((x, y)) for x in range(self.size)) for y in range(self.size)]

    def render(self) -> str:
        rows = []
        for y in range(self.size):
            row = []
            for x in range(self.size):
                here = (x, y) == self.agent
                row.append(AGENT_GLYPHS[self.direction] if here else self.cell((x, y)))
            rows.append(" ".join(row))
        return "\n".join(rows)

    # -------------------------------------------------------------- internals

    def _require_reset(self) -> None:
        if not self._ready:
            raise RuntimeError("call reset() before using the environment")

    def _observe(self) -> Observation:
        ahead = self.ahead()
        status = self._status()
        text = "\n".join(
            [
                f"Mission: {self.mission}.",
                "Map (x grows right, y grows down):",
                self.render(),
                f"Legend: {self.legend}, >v<^ you (pointing the way you face).",
                f"You are at ({self.agent[0]}, {self.agent[1]}) facing "
                f"{DIRECTION_NAMES[self.direction]}. Directly ahead: {self.cell(ahead)!r}."
                + (f" {status}" if status else ""),
                f"Step {self._steps} of {self.max_steps}.",
            ]
        )
        state = {
            "size": self.size,
            "agent": self.agent,
            "direction": self.direction,
            "goal": self.goal,
            "grid": self.grid(),
            **self._extra_state(),
        }
        return Observation(text=text, mission=self.mission, actions=self.actions, state=state)
