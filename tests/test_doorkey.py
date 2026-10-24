import pytest

from harness.envs import InvalidActionError, make_env
from harness.envs.doorkey import DoorKeyEnv


def test_reset_is_deterministic_per_seed() -> None:
    a, b = DoorKeyEnv(6), DoorKeyEnv(6)
    assert a.reset(7).text == b.reset(7).text
    layouts = {DoorKeyEnv(8).reset(s).text for s in range(20)}
    assert len(layouts) > 1, "different seeds should give different layouts"


def test_layout_invariants() -> None:
    for size in (5, 6, 8):
        for seed in range(30):
            env = DoorKeyEnv(size)
            s = env.reset(seed).state
            assert s["agent"][0] < s["wall_x"] and s["key"][0] < s["wall_x"]
            assert s["agent"] != s["key"]
            assert s["door"][0] == s["wall_x"]
            assert s["goal"] == (size - 2, size - 2)
            assert s["door_locked"] and not s["carrying_key"]


def test_locked_door_blocks_and_opens_only_with_key() -> None:
    env = DoorKeyEnv(5)
    env.reset(0)
    # Put the agent directly west of the door, facing it, with the key elsewhere.
    env.agent, env.direction = (env.door[0] - 1, env.door[1]), 0
    env.key = (1, 1 if env.door[1] != 1 else 2)
    env.step("forward")
    assert env.agent != env.door
    env.step("toggle")
    assert env.door_locked, "toggle without key must not unlock"
    env.carrying_key, env.key = True, None
    env.step("toggle")
    assert not env.door_locked and env.door_open
    env.step("forward")
    assert env.agent == env.door


def test_key_blocks_movement_until_picked_up() -> None:
    env = DoorKeyEnv(5)
    env.reset(0)
    env.agent, env.direction, env.key = (1, 1), 1, (1, 2)
    env.step("forward")
    assert env.agent == (1, 1)
    env.step("pickup")
    assert env.carrying_key and env.key is None
    env.step("forward")
    assert env.agent == (1, 2)


def test_invalid_action_raises() -> None:
    env = make_env("doorkey-5x5")
    env.reset(0)
    with pytest.raises(InvalidActionError):
        env.step("jump")


def test_truncates_at_max_steps() -> None:
    env = DoorKeyEnv(5)
    env.reset(0)
    result = None
    for _ in range(env.max_steps):
        result = env.step("left")
    assert result is not None and result.truncated and not result.terminated


def test_unknown_task() -> None:
    with pytest.raises(KeyError):
        make_env("nope")
