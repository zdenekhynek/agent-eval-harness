"""Task registry. A task id names one environment configuration."""

from __future__ import annotations

from collections.abc import Callable

from harness.envs.base import Env, InvalidActionError, Observation, StepResult
from harness.envs.doorkey import DoorKeyEnv
from harness.envs.grid import GridEnv
from harness.envs.navigation import EmptyEnv, LavaCrossingEnv, LavaGapEnv

_FAMILIES: dict[type[GridEnv], tuple[int, ...]] = {
    EmptyEnv: (5, 6, 8),
    LavaGapEnv: (5, 6, 7),
    DoorKeyEnv: (5, 6, 8),
}


def _factory(cls: type[GridEnv], size: int) -> Callable[[], Env]:
    return lambda: cls(size)


TASKS: dict[str, Callable[[], Env]] = {
    f"{cls.name}-{n}x{n}": _factory(cls, n) for cls, sizes in _FAMILIES.items() for n in sizes
}


def _crossing(rivers: int) -> Callable[[], Env]:
    return lambda: LavaCrossingEnv(9, rivers)


# LavaCrossing is named by river count (MiniGrid's S9N1-S9N3), always 9x9.
TASKS.update({f"lavacrossing-n{k}": _crossing(k) for k in (1, 2, 3)})


def make_env(task_id: str) -> Env:
    try:
        return TASKS[task_id]()
    except KeyError:
        raise KeyError(f"unknown task {task_id!r}; known: {sorted(TASKS)}") from None


__all__ = ["TASKS", "Env", "InvalidActionError", "Observation", "StepResult", "make_env"]
