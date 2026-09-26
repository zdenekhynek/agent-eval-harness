"""Task registry. A task id names one environment configuration."""

from __future__ import annotations

from collections.abc import Callable

from harness.envs.base import Env, InvalidActionError, Observation, StepResult
from harness.envs.doorkey import DoorKeyEnv

TASKS: dict[str, Callable[[], Env]] = {
    "doorkey-5x5": lambda: DoorKeyEnv(5),
    "doorkey-6x6": lambda: DoorKeyEnv(6),
    "doorkey-8x8": lambda: DoorKeyEnv(8),
}


def make_env(task_id: str) -> Env:
    try:
        return TASKS[task_id]()
    except KeyError:
        raise KeyError(f"unknown task {task_id!r}; known: {sorted(TASKS)}") from None


__all__ = ["TASKS", "Env", "InvalidActionError", "Observation", "StepResult", "make_env"]
