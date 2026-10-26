from pathlib import Path

import pytest

from harness import __version__
from harness.agents import Decision, make_agent
from harness.envs import TASKS, make_env
from harness.envs.base import Observation
from harness.runner import (
    MAX_CONSECUTIVE_INVALID,
    REFERENCE_SEEDS,
    RunConfigError,
    references,
    run_episode,
    run_matrix,
    run_references,
)
from harness.score import format_table, score
from harness.store import Store


@pytest.mark.parametrize("task", sorted(TASKS))
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


def test_resume_refuses_a_changed_step_cap(tmp_path: Path) -> None:
    with Store(tmp_path / "runs.db") as store:
        run_matrix(store, "r", ["scripted"], ["empty-5x5"], range(2), max_steps=20)
        with pytest.raises(RunConfigError, match="max-steps"):
            run_matrix(store, "r", ["scripted"], ["empty-5x5"], range(4), max_steps=30)


class _Prompted:
    """Stands in for an LLM agent: exposes a fingerprint like ``LLMAgent`` does."""

    name = "p"

    def __init__(self, fp: str) -> None:
        self.fp = fp

    def fingerprint(self) -> str:
        return self.fp

    def reset(self, task_id: str, seed: int) -> None:
        pass

    def act(self, observation: Observation) -> Decision:
        return Decision("forward")


def test_resume_refuses_a_changed_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("harness.runner.canonical_name", lambda spec: spec)
    with Store(tmp_path / "runs.db") as store:
        run_matrix(
            store, "r", ["p"], ["empty-5x5"], range(2), agent_factory=lambda _: _Prompted("v1")
        )
        # Same fingerprint resumes (and extends) fine.
        s = run_matrix(
            store, "r", ["p"], ["empty-5x5"], range(3), agent_factory=lambda _: _Prompted("v1")
        )
        assert (s.ran, s.skipped) == (1, 2)
        with pytest.raises(RunConfigError, match="prompt"):
            run_matrix(
                store, "r", ["p"], ["empty-5x5"], range(3), agent_factory=lambda _: _Prompted("v2")
            )
        config = store.run_config("r")
        assert config is not None and config["fingerprints"] == {"p": "v1"}
        assert config["seeds"] == [0, 1, 2]


def test_references_bracket_baselines_and_are_cached(tmp_path: Path) -> None:
    with Store(tmp_path / "runs.db") as store:
        refs = references(store, ["lavagap-5x5"])
        ref = refs["lavagap-5x5"]
        assert ref.floor < 0.2 < 0.9 < ref.ceiling and ref.n == len(REFERENCE_SEEDS)
        assert ref.normalise(ref.floor) == 0 and ref.normalise(ref.ceiling) == 1
        assert store.reference("lavagap-5x5", __version__) == ref


def test_references_follow_the_runs_step_cap(tmp_path: Path) -> None:
    with Store(tmp_path / "runs.db") as store:
        uncapped = references(store, ["doorkey-5x5"])["doorkey-5x5"]
        capped = references(store, ["doorkey-5x5"], max_steps=40)["doorkey-5x5"]
        # Random rarely finds the goal in 40 steps, so the floor drops; scripted
        # never needs that many, so the ceiling holds.
        assert capped.floor < uncapped.floor
        assert capped.ceiling == uncapped.ceiling

        run_matrix(store, "capped", ["random"], ["doorkey-5x5"], range(2), max_steps=40)
        assert run_references(store, "capped", ["doorkey-5x5"])["doorkey-5x5"] == capped
