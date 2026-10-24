import pytest

from harness.envs import make_env
from harness.envs.navigation import EmptyEnv, LavaGapEnv


@pytest.mark.parametrize("task", ["empty-8x8", "lavagap-7x7"])
def test_reset_is_deterministic_and_varies_by_seed(task: str) -> None:
    assert make_env(task).reset(4).text == make_env(task).reset(4).text
    assert len({make_env(task).reset(s).text for s in range(20)}) > 1


def test_empty_never_starts_on_the_goal() -> None:
    env = EmptyEnv(5)
    for seed in range(100):
        env.reset(seed)
        assert env.agent != env.goal


def test_lavagap_layout() -> None:
    for size in (5, 6, 7):
        env = LavaGapEnv(size)
        for seed in range(30):
            grid = env.reset(seed).state["grid"]
            column = [row[env.lava_x] for row in grid[1:-1]]
            assert column.count(".") == 1 and column.count("~") == size - 3
            assert 2 <= env.lava_x <= size - 3 and env.agent == (1, 1)


def test_stepping_on_lava_ends_the_episode_in_failure() -> None:
    env = LavaGapEnv(5)
    env.reset(0)
    env.agent, env.direction = (env.lava_x - 1, env.gap[1] % 3 + 1), 0
    assert env.cell(env.ahead()) == "~"
    result = env.step("forward")
    assert result.terminated and not result.success and result.reward == 0.0


def test_walking_through_the_gap_is_safe() -> None:
    env = LavaGapEnv(5)
    env.reset(0)
    env.agent, env.direction = (env.lava_x - 1, env.gap[1]), 0
    result = env.step("forward")
    assert env.agent == env.gap and not result.terminated


def test_reaching_the_goal_succeeds_with_minigrid_reward() -> None:
    env = EmptyEnv(5)
    env.reset(0)
    env.agent, env.direction = (env.goal[0] - 1, env.goal[1]), 0
    result = env.step("forward")
    assert result.terminated and result.success
    assert result.reward == pytest.approx(1 - 0.9 / env.max_steps)


def test_navigation_tasks_only_offer_movement() -> None:
    obs = make_env("empty-5x5").reset(0)
    assert obs.actions == ("left", "right", "forward")
    assert "lava" in make_env("lavagap-5x5").reset(0).text
