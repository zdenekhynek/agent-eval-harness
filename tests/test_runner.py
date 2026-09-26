from pathlib import Path

import pytest

from harness.agents import Decision, make_agent
from harness.envs import make_env
from harness.envs.base import Observation
from harness.runner import MAX_CONSECUTIVE_INVALID, run_episode, run_matrix
from harness.score import format_table, score
from harness.store import Store


@pytest.mark.parametrize("task", ["doorkey-5x5", "doorkey-6x6", "doorkey-8x8"])
def test_scripted_agent_solves_every_seed(task: str) -> None:
    env, agent = make_env(task), make_agent("scripted")
    for seed in range(50):
        ep = run_episode(agent, env, seed, "t")
        assert ep.success, f"{task} seed {seed} unsolved"
        assert ep.total_return > 0.9


def test_random_agent_is_reproducible() -> None:
    env = make_env("doorkey-5x5")
    a = run_episode(make_agent("random"), env, 3, "t")
    b = run_episode(make_agent("random"), env, 3, "t")
    assert [s.action for s in a.trace] == [s.action for s in b.trace]


class _Babbler:
    name = "babbler"

    def reset(self, task_id: str, seed: int) -> None:
        pass

    def act(self, observation: Observation) -> Decision:
        return Decision("jump", thought="I think I can fly")


def test_repeated_invalid_actions_stop_the_episode() -> None:
    ep = run_episode(_Babbler(), make_env("doorkey-5x5"), 0, "t")
    assert ep.invalid_actions == MAX_CONSECUTIVE_INVALID
    assert ep.truncated and not ep.success
    assert all(s.invalid for s in ep.trace)


def test_matrix_persists_and_resumes(tmp_path: Path) -> None:
    db = tmp_path / "runs.db"
    with Store(db) as store:
        first = run_matrix(store, "r1", ["random", "scripted"], ["doorkey-5x5"], range(5))
        assert (first.ran, first.skipped) == (10, 0)
    with Store(db) as store:
        again = run_matrix(store, "r1", ["random", "scripted"], ["doorkey-5x5"], range(8))
        assert (again.ran, again.skipped) == (6, 10)
        assert len(store.episodes("r1")) == 16
        ep = store.episode("r1", "scripted", "doorkey-5x5", 2)
        assert ep is not None and ep.success and len(ep.trace) == ep.steps


def test_scores_rank_scripted_above_random(tmp_path: Path) -> None:
    with Store(tmp_path / "runs.db") as store:
        run_matrix(store, "r", ["random", "scripted"], ["doorkey-5x5"], range(20))
        scores = score(store.episodes("r"))
    assert [s.agent for s in scores] == ["scripted", "random"]
    assert scores[0].success_rate == 1.0
    assert scores[1].success_rate < 1.0
    assert "scripted" in format_table(scores)
