import pytest

from harness.score import (
    Reference,
    bootstrap_ci,
    compare,
    format_compare,
    format_overall,
    format_table,
    overall,
    score,
    stratified_bootstrap_ci,
    wilson,
)
from harness.trace import Episode

REFS = {"a": Reference(floor=0.0, ceiling=1.0, n=10), "b": Reference(0.2, 0.6, n=10)}


def _ep(agent: str, task: str, seed: int, ret: float) -> Episode:
    return Episode("r", agent, task, seed, success=ret > 0, steps=10, total_return=ret)


def test_wilson_matches_known_values() -> None:
    lo, hi = wilson(0, 10)
    assert lo == 0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    assert wilson(10, 10)[1] == 1.0


def test_bootstrap_is_reproducible_and_brackets_the_mean() -> None:
    values = [0.0, 1.0] * 20
    ci = bootstrap_ci(values)
    assert ci == bootstrap_ci(values)
    assert ci[0] < 0.5 < ci[1]
    assert bootstrap_ci([0.3] * 5) == (0.3, 0.3)


def test_stratified_bootstrap_weights_groups_equally() -> None:
    # A big group of 0s and a small group of 1s: equal weight means ~0.5, not ~0.1.
    lo, hi = stratified_bootstrap_ci([[0.0] * 90, [1.0] * 10])
    assert lo == hi == pytest.approx(0.5)


def test_reference_normalises_and_handles_degenerate_span() -> None:
    assert REFS["b"].normalise(0.4) == pytest.approx(0.5)
    assert Reference(0.5, 0.5, 1).normalise(0.7) is None


def test_score_adds_intervals_and_normalised_score() -> None:
    eps = [_ep("x", "b", s, 0.6 if s % 2 else 0.2) for s in range(10)]
    (s,) = score(eps, REFS)
    assert s.success_rate == 1.0 and s.success_ci[1] == 1.0
    assert s.norm == pytest.approx(0.5)
    assert s.norm_ci is not None and s.norm_ci[0] < 0.5 < s.norm_ci[1]
    (bare,) = score(eps)
    assert bare.norm is None and "-" in format_table([bare])


def test_overall_needs_every_task() -> None:
    eps = [_ep("full", t, s, 1.0) for t in "ab" for s in range(3)]
    eps += [_ep("partial", "a", s, 1.0) for s in range(3)]
    rows = overall(eps, REFS)
    assert [r.agent for r in rows] == ["full"]
    assert rows[0].tasks == 2 and rows[0].n == 6
    assert "full" in format_overall(rows)
    assert overall(eps, {"a": REFS["a"]}) == []


def test_compare_pairs_on_seeds() -> None:
    eps = [_ep("good", t, s, 0.6 if t == "b" else 0.9) for t in "ab" for s in range(20)]
    eps += [_ep("bad", t, s, 0.2 if t == "b" else 0.1) for t in "ab" for s in range(15)]
    rows = compare(eps, "good", "bad", REFS)
    assert [r.task for r in rows] == ["a", "b", "all"]
    assert all(r.pairs == 15 for r in rows[:2]) and rows[2].pairs == 30
    assert rows[0].diff == pytest.approx(0.8) and rows[1].diff == pytest.approx(1.0)
    assert {r.verdict for r in rows} == {"A better"}
    assert compare(eps, "bad", "good", REFS)[0].verdict == "B better"


def test_compare_falls_back_to_success_rate_and_can_be_inconclusive() -> None:
    eps = [_ep("x", "c", s, s % 2) for s in range(10)]
    eps += [_ep("y", "c", s, (s + 1) % 2) for s in range(10)]
    (row,) = compare(eps, "x", "y", REFS)
    assert row.metric == "success" and row.verdict == "no clear difference"
    assert "success rate" in format_compare([row], "x", "y")
    assert "no seeds" in format_compare(compare(eps, "x", "z"), "x", "z")


def test_tiny_but_real_differences_are_marginal_not_wins() -> None:
    # Every seed differs by exactly 0.01: the CI excludes zero, but the effect is tiny.
    eps = [_ep("a", "a", s, 0.91) for s in range(10)]
    eps += [_ep("b", "a", s, 0.90) for s in range(10)]
    (row,) = compare(eps, "a", "b", REFS)
    assert row.diff_ci[0] > 0
    assert row.status == "marginal" and "marginally" in row.verdict
    (strict,) = compare(eps, "a", "b", REFS, min_effect=0.005)
    assert strict.status == "ahead"
    (flipped,) = compare(eps, "b", "a", REFS)
    assert flipped.status == "marginal" and flipped.verdict.startswith("B")
