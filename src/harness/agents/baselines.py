"""Baselines that bracket every LLM agent: a random floor and a scripted ceiling."""

from __future__ import annotations

import random
from collections import deque

from harness.agents.base import Decision
from harness.envs.base import Observation
from harness.envs.grid import DIRECTIONS

Pos = tuple[int, int]

# Cells the scripted agent will walk through. Lava is enterable but never safe.
SAFE = frozenset({".", "G", "_"})


class RandomAgent:
    """Uniformly random actions, seeded per episode so runs are reproducible."""

    name = "random"

    def __init__(self) -> None:
        self._rng = random.Random(0)

    def reset(self, task_id: str, seed: int) -> None:
        self._rng = random.Random(f"random:{task_id}:{seed}")

    def act(self, observation: Observation) -> Decision:
        return Decision(self._rng.choice(observation.actions))


class ScriptedAgent:
    """Near-optimal gridworld solver that reads privileged state.

    It replans every step with a breadth-first search over (position,
    direction) on the glyph grid, avoiding lava: face the key and pick it up,
    face the door and unlock it, then walk onto the goal. It is the ceiling an
    LLM agent is measured against, not a competitor - it cheats by reading
    ``observation.state``.
    """

    name = "scripted"

    def reset(self, task_id: str, seed: int) -> None:
        pass

    def act(self, observation: Observation) -> Decision:
        s = observation.state
        if "grid" not in s:
            raise ValueError("scripted agent needs a gridworld observation")
        grid: list[str] = s["grid"]
        keys, doors = _find(grid, "K"), _find(grid, "L") + _find(grid, "D")
        if keys and not s.get("carrying_key"):
            target, final, stand_on, thought = keys[0], "pickup", False, "get the key"
        elif doors:
            target, final, stand_on, thought = doors[0], "toggle", False, "open the door"
        else:
            target, final, stand_on, thought = _find(grid, "G")[0], "forward", True, "go to goal"

        ax, ay = s["agent"]
        plan = _plan(grid, ((int(ax), int(ay)), int(s["direction"])), target, stand_on)
        if plan is None:
            return Decision("left", thought=f"{thought}: no path found, turning")
        if not plan:
            return Decision(final, thought=thought)
        return Decision(plan[0], thought=thought)


def _find(grid: list[str], glyph: str) -> list[Pos]:
    return [(x, y) for y, row in enumerate(grid) for x, c in enumerate(row) if c == glyph]


def _plan(grid: list[str], start: tuple[Pos, int], target: Pos, stand_on: bool) -> list[str] | None:
    """Shortest turn/move sequence until we are facing ``target``.

    With ``stand_on`` the returned plan stops one ``forward`` short of standing
    on it, so the caller's final action (``forward``) completes it.
    """

    def facing(state: tuple[Pos, int]) -> Pos:
        (x, y), d = state
        dx, dy = DIRECTIONS[d]
        return (x + dx, y + dy)

    def safe(pos: Pos) -> bool:
        x, y = pos
        return 0 <= y < len(grid) and 0 <= x < len(grid[y]) and grid[y][x] in SAFE

    queue: deque[tuple[tuple[Pos, int], list[str]]] = deque([(start, [])])
    seen = {start}
    while queue:
        state, path = queue.popleft()
        if facing(state) == target:
            return path
        (x, y), d = state
        moves: list[tuple[str, tuple[Pos, int]]] = [
            ("left", ((x, y), (d - 1) % 4)),
            ("right", ((x, y), (d + 1) % 4)),
        ]
        ahead = facing(state)
        if safe(ahead) and not (stand_on and ahead == target):
            moves.append(("forward", (ahead, d)))
        for action, nxt in moves:
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, [*path, action]))
    return None
