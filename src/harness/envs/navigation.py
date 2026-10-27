"""Navigation tasks with MiniGrid semantics: ``Empty-Random`` and ``LavaGap``.

Together with DoorKey they separate skills: ``empty`` is pure spatial
navigation (the floor any agent should clear), ``lavagap`` adds a hazard -
stepping onto lava ends the episode with zero reward, so it measures whether
an agent reads the map before it moves - and ``doorkey`` adds multi-step
planning.
"""

from __future__ import annotations

import random

from harness.envs.grid import LAVA, GridEnv, Pos


class EmptyEnv(GridEnv):
    """``MiniGrid-Empty-Random-NxN``: an empty room, random start pose, goal in the corner."""

    name = "empty"
    mission = "get to the goal square"

    def _generate(self, rng: random.Random) -> None:
        n = self.size
        cells = [(x, y) for x in range(1, n - 1) for y in range(1, n - 1)]
        self.agent = rng.choice([c for c in cells if c != self.goal])
        self.direction = rng.randrange(4)


class LavaGapEnv(GridEnv):
    """``MiniGrid-LavaGapS<N>``: a lava column with one gap between start and goal.

    The agent always starts at (1, 1) facing east; the column's x and the gap's
    y vary with the seed. Small rooms have few distinct layouts (three at 5x5),
    so seeds repeat layouts - fine for estimating the task's success rate, but
    report the size alongside the score.
    """

    name = "lavagap"
    mission = "avoid the lava and get to the goal square"
    legend = "# wall, . floor, ~ lava (stepping on it ends the episode in failure), G goal"

    def __init__(self, size: int = 5) -> None:
        super().__init__(size)
        self.lava_x = 2
        self.gap: Pos = (2, 1)

    def _generate(self, rng: random.Random) -> None:
        n = self.size
        self.lava_x = rng.randrange(2, n - 2)
        self.gap = (self.lava_x, rng.randrange(1, n - 1))
        self.agent, self.direction = (1, 1), 0

    def cell(self, pos: Pos) -> str:
        x, y = pos
        if x == self.lava_x and 0 < y < self.size - 1 and pos != self.gap:
            return LAVA
        return super().cell(pos)


class LavaCrossingEnv(GridEnv):
    """``MiniGrid-LavaCrossingS9N<k>``: ``k`` lava rivers, each with one opening.

    Rivers run along even rows or columns and split the room into a grid of
    chambers; the openings are cut along a random monotone path from the
    agent's corner to the goal, so a safe route always exists but has to be
    found. Same generation as MiniGrid's ``CrossingEnv``. Harder than LavaGap:
    longer routes, several hazards, and openings that are not in line.
    """

    name = "lavacrossing"
    mission = "avoid the lava and get to the goal square"
    legend = LavaGapEnv.legend

    def __init__(self, size: int = 9, crossings: int = 1) -> None:
        super().__init__(size)
        self.crossings = crossings
        self.task_id = f"{self.name}-n{crossings}"
        self.lava: set[Pos] = set()

    def _generate(self, rng: random.Random) -> None:
        n = self.size
        self.agent, self.direction = (1, 1), 0
        # A vertical river sits at an even x; a horizontal one at an even y.
        rivers = [("v", i) for i in range(2, n - 2, 2)] + [("h", j) for j in range(2, n - 2, 2)]
        rng.shuffle(rivers)
        rivers = rivers[: self.crossings]
        xs = sorted(pos for kind, pos in rivers if kind == "v")
        ys = sorted(pos for kind, pos in rivers if kind == "h")
        self.lava = {(x, y) for x in xs for y in range(1, n - 1)}
        self.lava |= {(x, y) for y in ys for x in range(1, n - 1)}

        # Cut one opening per river along a random monotone path of chambers.
        path = ["right"] * len(xs) + ["down"] * len(ys)
        rng.shuffle(path)
        bounds_x, bounds_y = [0, *xs, n - 1], [0, *ys, n - 1]
        room_x = room_y = 0
        for step in path:
            if step == "right":
                x = bounds_x[room_x + 1]
                y = rng.randrange(bounds_y[room_y] + 1, bounds_y[room_y + 1])
                room_x += 1
            else:
                x = rng.randrange(bounds_x[room_x] + 1, bounds_x[room_x + 1])
                y = bounds_y[room_y + 1]
                room_y += 1
            self.lava.discard((x, y))

    def cell(self, pos: Pos) -> str:
        return LAVA if pos in self.lava else super().cell(pos)
