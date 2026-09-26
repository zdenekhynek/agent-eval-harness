"""LLM agent + API client tests. A fake transport replays canned API responses,
so these run offline with no key. ``test_live_*`` hits the real API and only
runs when ANTHROPIC_API_KEY is set and HARNESS_LIVE=1."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from harness.agents import canonical_name, make_agent, parse_spec
from harness.agents.llm import NO_ACTION
from harness.cli import main
from harness.envs import make_env
from harness.llmapi import (
    AnthropicClient,
    APIError,
    MissingAPIKeyError,
    Usage,
    cost_usd,
)
from harness.runner import run_episode, run_matrix
from harness.score import score
from harness.store import Store


def _ok(body: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
    return 200, {}, json.dumps(body).encode()


def _reply(action: str | None, text: str = "", tin: int = 400, tout: int = 30) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    if action is not None:
        content.append({"type": "tool_use", "id": "t", "name": "act", "input": {"action": action}})
    return {
        "content": content,
        "model": "claude-sonnet-5",
        "stop_reason": "tool_use" if action else "end_turn",
        "usage": {"input_tokens": tin, "output_tokens": tout},
    }


class FakeTransport:
    """Replays responses in order and records every request payload."""

    def __init__(self, responses: list[tuple[int, dict[str, str], bytes]]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []

    def __call__(
        self, url: str, headers: dict[str, str], body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        self.requests.append(json.loads(body))
        self.headers.append(headers)
        return self.responses.pop(0)


def _client(transport: FakeTransport) -> AnthropicClient:
    return AnthropicClient(api_key="test-key", transport=transport, sleep=lambda _: None)


def _scripted_actions(task: str, seed: int) -> list[str]:
    ep = run_episode(make_agent("scripted"), make_env(task), seed, "t")
    assert ep.success
    return [s.action for s in ep.trace]


# ------------------------------------------------------------------ the agent


def test_react_agent_solves_doorkey_and_records_usage() -> None:
    actions = _scripted_actions("doorkey-5x5", 0)
    transport = FakeTransport([_ok(_reply(a, text=f"I should {a}.")) for a in actions])
    agent = make_agent("react", client=_client(transport))

    ep = run_episode(agent, make_env("doorkey-5x5"), 0, "t")

    assert ep.success and ep.agent == "react:claude-sonnet-5"
    assert ep.steps == len(actions)
    assert ep.input_tokens == 400 * len(actions) and ep.output_tokens == 30 * len(actions)
    assert ep.cost_usd == pytest.approx(len(actions) * (400 * 2 + 30 * 10) / 1e6)
    assert ep.trace[0].thought == f"I should {actions[0]}."
    assert ep.trace[0].cost_usd is not None and ep.trace[0].model_latency_ms is not None

    first, second = transport.requests[0], transport.requests[1]
    assert first["model"] == "claude-sonnet-5"
    assert first["tool_choice"] == {"type": "auto"}
    assert first["tools"][0]["input_schema"]["properties"]["action"]["enum"] == [
        "left",
        "right",
        "forward",
        "pickup",
        "toggle",
    ]
    assert "think briefly" in first["system"]
    assert "first step" in first["messages"][0]["content"]
    assert "Map" in first["messages"][0]["content"]
    assert f"step 0: {actions[0]}" in second["messages"][0]["content"]
    assert transport.headers[0]["x-api-key"] == "test-key"


def test_zero_shot_forces_the_tool() -> None:
    transport = FakeTransport([_ok(_reply("left"))])
    agent = make_agent("zero-shot:claude-haiku-4-5", client=_client(transport))
    agent.reset("doorkey-5x5", 0)
    decision = agent.act(make_env("doorkey-5x5").reset(0))
    assert decision.action == "left" and decision.thought is None
    req = transport.requests[0]
    assert req["tool_choice"] == {"type": "tool", "name": "act"}
    assert req["model"] == "claude-haiku-4-5"
    assert "think briefly" not in req["system"]


def test_memory_window_is_bounded() -> None:
    transport = FakeTransport([_ok(_reply("left")) for _ in range(5)])
    agent = make_agent("react", client=_client(transport), memory=2)
    env = make_env("doorkey-5x5")
    agent.reset(env.task_id, 0)
    obs = env.reset(0)
    for _ in range(5):
        agent.act(obs)
    prompt = transport.requests[-1]["messages"][0]["content"]
    assert "step 3: left" in prompt and "step 2: left" in prompt
    assert "step 1:" not in prompt


def test_reply_without_tool_call_is_an_invalid_action() -> None:
    transport = FakeTransport([_ok(_reply(None, text="Hmm, let me think.")) for _ in range(5)])
    ep = run_episode(
        make_agent("react", client=_client(transport)), make_env("doorkey-5x5"), 0, "t"
    )
    assert ep.invalid_actions == 5 and ep.truncated and not ep.success
    assert ep.trace[0].action == NO_ACTION
    assert ep.cost_usd is not None and ep.cost_usd > 0, "invalid steps still cost money"


def test_max_steps_caps_llm_episodes() -> None:
    transport = FakeTransport([_ok(_reply("left")) for _ in range(3)])
    agent = make_agent("zero-shot", client=_client(transport))
    ep = run_episode(agent, make_env("doorkey-8x8"), 0, "t", max_steps=3)
    assert ep.steps == 3 and ep.truncated and not transport.responses


# ----------------------------------------------------------------- the client


def test_client_retries_overloaded_then_succeeds() -> None:
    sleeps: list[float] = []
    transport = FakeTransport(
        [
            (529, {"retry-after": "3"}, b'{"error": {"message": "overloaded"}}'),
            (500, {}, b"oops"),
            _ok(_reply("forward")),
        ]
    )
    client = AnthropicClient(api_key="k", transport=transport, sleep=sleeps.append)
    resp = client.messages(model="claude-sonnet-5", system="s", messages=[], max_tokens=10)
    assert resp.retries == 2 and sleeps == [3.0, 2.0]
    assert resp.tool_uses()[0]["input"] == {"action": "forward"}


def test_client_raises_on_client_errors_and_exhausted_retries() -> None:
    bad = FakeTransport([(400, {}, b'{"error": {"message": "bad model"}}')])
    with pytest.raises(APIError, match="bad model"):
        _client(bad).messages(model="x", system="s", messages=[], max_tokens=10)

    busy = FakeTransport([(529, {}, b"{}")] * 3)
    client = AnthropicClient(api_key="k", transport=busy, max_retries=2, sleep=lambda _: None)
    with pytest.raises(APIError) as err:
        client.messages(model="x", system="s", messages=[], max_tokens=10)
    assert err.value.status == 529


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        AnthropicClient()


def test_cost_accounting() -> None:
    assert cost_usd("claude-sonnet-5", Usage(1_000_000, 100_000)) == pytest.approx(3.0)
    assert cost_usd("claude-haiku-4-5-20251001", Usage(1_000_000, 0)) == pytest.approx(1.0)
    cached = Usage(input_tokens=0, output_tokens=0, cache_read_input_tokens=1_000_000)
    assert cost_usd("claude-sonnet-5", cached) == pytest.approx(0.2)
    assert cost_usd("some-unpriced-model", Usage(10, 10)) is None


# ----------------------------------------------------- registry, runner, store


def test_agent_specs() -> None:
    assert parse_spec("react") == ("react", "claude-sonnet-5")
    assert parse_spec("react:claude-haiku-4-5") == ("react", "claude-haiku-4-5")
    assert canonical_name("zero-shot", "claude-haiku-4-5") == "zero-shot:claude-haiku-4-5"
    assert canonical_name("scripted") == "scripted"
    with pytest.raises(KeyError):
        parse_spec("scripted:claude-sonnet-5")
    with pytest.raises(KeyError):
        parse_spec("chain-of-vibes")


def test_run_stops_at_cost_cap_and_resumes(tmp_path: Path) -> None:
    def factory(spec: str) -> Any:
        transport = FakeTransport([_ok(_reply("left", tin=100_000, tout=0)) for _ in range(100)])
        return make_agent(spec, client=_client(transport))

    with Store(tmp_path / "r.db") as store:
        # each episode = 4 steps x 100k input tokens x $2/M = $0.80
        summary = run_matrix(
            store, "r", ["zero-shot"], ["doorkey-5x5"], range(5),
            agent_factory=factory, max_steps=4, max_cost_usd=1.0,
        )  # fmt: skip
        assert summary.stopped_for_budget and summary.ran == 2
        assert summary.cost_usd == pytest.approx(1.6)

        resumed = run_matrix(
            store, "r", ["zero-shot"], ["doorkey-5x5"], range(5),
            agent_factory=factory, max_steps=4,
        )  # fmt: skip
        assert (resumed.ran, resumed.skipped) == (3, 2)
        (s,) = score(store.episodes("r"))
        assert s.agent == "zero-shot:claude-sonnet-5" and s.n == 5
        assert s.mean_cost_usd == pytest.approx(0.8) and s.mean_tokens == 400_000
        ep = store.episode("r", "zero-shot:claude-sonnet-5", "doorkey-5x5", 0)
        assert ep is not None and ep.trace[0].input_tokens == 100_000


def test_store_migrates_milestone0_databases(tmp_path: Path) -> None:
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """CREATE TABLE runs (id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
               config_json TEXT NOT NULL);
           CREATE TABLE episodes (run_id TEXT, agent TEXT, task TEXT, seed INTEGER,
               success INTEGER, steps INTEGER, total_return REAL, truncated INTEGER,
               invalid_actions INTEGER, duration_ms REAL, final_observation TEXT,
               trace_json TEXT, PRIMARY KEY (run_id, agent, task, seed));
           INSERT INTO runs VALUES ('old', '2026-09-26T09:00:00', '{}');
           INSERT INTO episodes VALUES ('old','random','doorkey-5x5',0,0,3,0.0,1,0,1.0,'',
               '[{"t":0,"observation":"o","action":"left","reward":0.0}]');"""
    )
    conn.commit()
    conn.close()
    with Store(db) as store:
        ep = store.episode("old", "random", "doorkey-5x5", 0)
        assert ep is not None and ep.cost_usd is None and ep.input_tokens == 0
        assert ep.trace[0].model_latency_ms is None


def test_cli_needs_a_key_only_for_llm_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db = str(tmp_path / "x.db")
    assert main(["--db", db, "run", "--agents", "react", "--seeds", "1"]) == 2
    assert main(["--db", db, "run", "--agents", "scripted", "--seeds", "1"]) == 0
    assert main(["--db", db, "run", "--agents", "nope"]) == 2


# ----------------------------------------------------------------------- live


@pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("HARNESS_LIVE") == "1"),
    reason="live API test: set ANTHROPIC_API_KEY and HARNESS_LIVE=1",
)
def test_live_react_makes_a_valid_move() -> None:
    agent = make_agent("react:claude-haiku-4-5")
    env = make_env("doorkey-5x5")
    agent.reset(env.task_id, 0)
    decision = agent.act(env.reset(0))
    assert decision.action in env.reset(0).actions
    assert decision.input_tokens and decision.cost_usd
