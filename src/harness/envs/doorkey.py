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
from typing import Any

from harness.envs.grid import GridEnv, Pos

MISSION = "use the key to open the door and then get to the goal"


class DoorKeyEnv(GridEnv):
    """``DoorKey`` on an ``size`` x ``size`` grid (outer walls included)."""

    name = "doorkey"
    mission = MISSION
    legend = "# wall, . floor, K key, L locked door, D closed door, _ open door, G goal"
    actions = ("left", "right", "forward", "pickup", "toggle")
    steps_per_cell = 10

    def __init__(self, size: int = 5) -> None:
        super().__init__(size)
        self.wall_x = 2
        self.door: Pos = (2, 1)
        self.key: Pos | None = None  # None once picked up
        self.carrying_key = False
        self.door_locked = True
        self.door_open = False

    def _generate(self, rng: random.Random) -> None:
        n = self.size
        self.wall_x = rng.randrange(2, n - 2)
        self.door = (self.wall_x, rng.randrange(1, n - 1))
        left_cells = [(x, y) for x in range(1, self.wall_x) for y in range(1, n - 1)]
        self.agent = rng.choice(left_cells)
        self.key = rng.choice([c for c in left_cells if c != self.agent])
        self.direction = rng.randrange(4)
        self.carrying_key = False
        self.door_locked = True
        self.door_open = False

    def cell(self, pos: Pos) -> str:
        if pos == self.door:
            return "L" if self.door_locked else ("_" if self.door_open else "D")
        if pos[0] == self.wall_x:
            return "#"
        if pos == self.key:
            return "K"
        return super().cell(pos)

    def _interact(self, action: str, ahead: Pos) -> None:
        if action == "pickup":
            if self.key is not None and ahead == self.key and not self.carrying_key:
                self.key = None
                self.carrying_key = True
        elif action == "toggle" and ahead == self.door:
            if self.door_locked:
                if self.carrying_key:
                    self.door_locked = False
                    self.door_open = True
            else:
                self.door_open = not self.door_open

    def _status(self) -> str:
        return f"Carrying key: {'yes' if self.carrying_key else 'no'}."

    def _extra_state(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "carrying_key": self.carrying_key,
            "door": self.door,
            "door_locked": self.door_locked,
            "door_open": self.door_open,
            "wall_x": self.wall_x,
        }
