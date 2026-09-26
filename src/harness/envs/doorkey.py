"""DoorKey: a dependency-free gridworld with MiniGrid ``DoorKey`` semantics.

The agent starts in a room sealed by a wall with a locked door. It must pick
up the key, unlock the door and walk to the goal square in the bottom-right
corner. Solving it takes planning (key before door) and spatial reasoning
(turning to face things), which makes it a good first task for LLM agents.

Semantics follow MiniGrid's ``MiniGrid-DoorKey-NxN-v0``: the same action
names, turn-then-move movement, keys block movement until picked up, and a
success reward of ``1 - 0.9 * steps / max_steps``. It is a native
reimplementation rather than an adapter so the harness runs with no heavy
dependencies; a real MiniGrid adapter can sit behind the same ``Env`` protocol.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from harness.envs.base import InvalidActionError, Observation, StepResult

ACTIONS: tuple[str, ...] = ("left", "right", "forward", "pickup", "toggle")

# Direction index -> (dx, dy). 0 = east, 1 = south, 2 = west, 3 = north,
# matching MiniGrid's convention (y grows downwards).
DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))
DIRECTION_NAMES: tuple[str, ...] = ("east", "south", "west", "north")
AGENT_GLYPHS: tuple[str, ...] = (">", "v", "<", "^")

MISSION = "use the key to open the door and then get to the goal"

Pos = tuple[int, int]


@dataclass
class _State:
    agent: Pos
    direction: int
    key: Pos | None  # None once picked up
    carrying_key: bool
    door: Pos
    door_locked: bool
    door_open: bool
    goal: Pos
    wall_x: int


class DoorKeyEnv:
    """``DoorKey`` on an ``size`` x ``size`` grid (outer walls included)."""

    def __init__(self, size: int = 5) -> None:
        if size < 5:
            raise ValueError("DoorKey needs size >= 5")
        self.size = size
        self.task_id = f"doorkey-{size}x{size}"
        self.max_steps = 10 * size * size
        self._s: _State | None = None
        self._steps = 0

    # ------------------------------------------------------------------ setup

    def reset(self, seed: int) -> Observation:
        rng = random.Random(f"{self.task_id}:{seed}")
        n = self.size
        wall_x = rng.randrange(2, n - 2)
        door = (wall_x, rng.randrange(1, n - 1))
        left_cells = [(x, y) for x in range(1, wall_x) for y in range(1, n - 1)]
        agent = rng.choice(left_cells)
        key = rng.choice([c for c in left_cells if c != agent])
        self._s = _State(
            agent=agent,
            direction=rng.randrange(4),
            key=key,
            carrying_key=False,
            door=door,
            door_locked=True,
            door_open=False,
            goal=(n - 2, n - 2),
            wall_x=wall_x,
        )
        self._steps = 0
        return self._observe()

    # ------------------------------------------------------------------- step

    def step(self, action: str) -> StepResult:
        s = self._state()
        if action not in ACTIONS:
            raise InvalidActionError(f"unknown action {action!r}; expected one of {ACTIONS}")

        self._steps += 1
        reward = 0.0
        terminated = False
        ahead = self._ahead()

        if action == "left":
            s.direction = (s.direction - 1) % 4
        elif action == "right":
            s.direction = (s.direction + 1) % 4
        elif action == "forward":
            if self.passable(ahead):
                s.agent = ahead
                if ahead == s.goal:
                    terminated = True
                    reward = 1.0 - 0.9 * (self._steps / self.max_steps)
        elif action == "pickup":
            if s.key is not None and ahead == s.key and not s.carrying_key:
                s.key = None
                s.carrying_key = True
        elif action == "toggle" and ahead == s.door:
            if s.door_locked:
                if s.carrying_key:
                    s.door_locked = False
                    s.door_open = True
            else:
                s.door_open = not s.door_open

        truncated = not terminated and self._steps >= self.max_steps
        return StepResult(self._observe(), reward, terminated, truncated)

    # ---------------------------------------------------------------- queries

    def passable(self, pos: Pos) -> bool:
        """Can the agent stand on ``pos`` given the current door/key state?"""
        s = self._state()
        x, y = pos
        n = self.size
        if x <= 0 or y <= 0 or x >= n - 1 or y >= n - 1:
            return False
        if pos == s.door:
            return s.door_open
        if x == s.wall_x:
            return False
        return pos != s.key

    def cell(self, pos: Pos) -> str:
        """Glyph for ``pos``, ignoring the agent."""
        s = self._state()
        x, y = pos
        n = self.size
        if pos == s.door:
            return "L" if s.door_locked else ("_" if s.door_open else "D")
        if x in (0, n - 1) or y in (0, n - 1) or x == s.wall_x:
            return "#"
        if pos == s.key:
            return "K"
        if pos == s.goal:
            return "G"
        return "."

    def render(self) -> str:
        s = self._state()
        rows = []
        for y in range(self.size):
            row = []
            for x in range(self.size):
                row.append(AGENT_GLYPHS[s.direction] if (x, y) == s.agent else self.cell((x, y)))
            rows.append(" ".join(row))
        return "\n".join(rows)

    # -------------------------------------------------------------- internals

    def _state(self) -> _State:
        if self._s is None:
            raise RuntimeError("call reset() before using the environment")
        return self._s

    def _ahead(self) -> Pos:
        s = self._state()
        dx, dy = DIRECTIONS[s.direction]
        return (s.agent[0] + dx, s.agent[1] + dy)

    def _observe(self) -> Observation:
        s = self._state()
        ahead = self._ahead()
        text = "\n".join(
            [
                f"Mission: {MISSION}.",
                "Map (x grows right, y grows down):",
                self.render(),
                "Legend: # wall, . floor, K key, L locked door, D closed door, "
                "_ open door, G goal, >v<^ you (pointing the way you face).",
                f"You are at ({s.agent[0]}, {s.agent[1]}) facing {DIRECTION_NAMES[s.direction]}. "
                f"Directly ahead: {self.cell(ahead)!r}. "
                f"Carrying key: {'yes' if s.carrying_key else 'no'}.",
                f"Step {self._steps} of {self.max_steps}.",
            ]
        )
        state = {
            "size": self.size,
            "agent": s.agent,
            "direction": s.direction,
            "key": s.key,
            "carrying_key": s.carrying_key,
            "door": s.door,
            "door_locked": s.door_locked,
            "door_open": s.door_open,
            "goal": s.goal,
            "wall_x": s.wall_x,
        }
        return Observation(text=text, mission=MISSION, actions=ACTIONS, state=state)
