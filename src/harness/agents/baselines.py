"""Baselines that bracket every LLM agent: a random floor and a scripted ceiling."""

from __future__ import annotations

import random
from collections import deque
from typing import Any

from harness.agents.base import Decision
from harness.envs.base import Observation
from harness.envs.doorkey import DIRECTIONS

Pos = tuple[int, int]


class RandomAgent:
    """Uniformly random actions, seeded per episode so runs are reproducible."""

    name = "random"

    def __init__(self) -> None:
        self._rng = random.Random(0)

    def reset(self, task_id: str, seed: int) -> None:
        self._rng = random.Random(f"random:{task_id}:{seed}")

    def act(self, observation: Observation) -> Decision:
        return Decision(self._rng.choice(observation.actions))


class ScriptedDoorKeyAgent:
    """Optimal-ish DoorKey solver that reads privileged state.

    It replans every step with a breadth-first search over (position,
    direction): face the key and pick it up, face the door and unlock it,
    then walk onto the goal. It is the ceiling an LLM agent is measured
    against, not a competitor - it cheats by reading ``observation.state``.
    """

    name = "scripted"

    def reset(self, task_id: str, seed: int) -> None:
        if not task_id.startswith("doorkey-"):
            raise ValueError(f"scripted agent only knows DoorKey tasks, got {task_id!r}")

    def act(self, observation: Observation) -> Decision:
        s = observation.state
        if not s["carrying_key"] and s["key"] is not None:
            target, final, goal_is_standing = s["key"], "pickup", False
            thought = "get the key"
        elif not s["door_open"]:
            target, final, goal_is_standing = s["door"], "toggle", False
            thought = "open the door"
        else:
            target, final, goal_is_standing = s["goal"], "forward", True
            thought = "walk to the goal"

        plan = _plan(s, tuple(target), goal_is_standing)
        if plan is None:
            return Decision("left", thought=f"{thought}: no path found, turning")
        if not plan:
            return Decision(final, thought=thought)
        return Decision(plan[0], thought=thought)


def _passable(s: dict[str, Any], pos: Pos) -> bool:
    n, (x, y) = s["size"], pos
    if x <= 0 or y <= 0 or x >= n - 1 or y >= n - 1:
        return False
    if pos == tuple(s["door"]):
        return bool(s["door_open"])
    if x == s["wall_x"]:
        return False
    return s["key"] is None or pos != tuple(s["key"])


def _plan(s: dict[str, Any], target: Pos, stand_on: bool) -> list[str] | None:
    """Shortest turn/move sequence until we are facing ``target``.

    With ``stand_on`` the returned plan stops one ``forward`` short of standing
    on it, so the caller's final action (``forward``) completes it.
    """
    start = (tuple(s["agent"]), int(s["direction"]))

    def facing(state: tuple[Pos, int]) -> Pos:
        (x, y), d = state
        dx, dy = DIRECTIONS[d]
        return (x + dx, y + dy)

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
        if _passable(s, ahead) and not (stand_on and ahead == target):
            moves.append(("forward", (ahead, d)))
        for action, nxt in moves:
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, [*path, action]))
    return None
