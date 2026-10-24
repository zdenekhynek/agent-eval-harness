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
