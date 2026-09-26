# agent-eval-harness

**Which prompting strategy actually makes an LLM agent smarter — and how would you know?**

A small, honest harness for measuring LLM agents: run agents across a suite of
tasks and seeds, record every step (observation, reasoning, action, reward,
tokens, cost), score them with proper statistics, and explore the results.

> Status: **milestone 1 — LLM agents.** Zero-shot and ReAct agents on Claude,
> with tokens, cost and latency recorded per step. Confidence intervals next.

## Quick start

```bash
uv sync --group dev
uv run harness run --tasks doorkey-5x5,doorkey-8x8 --seeds 50
uv run harness show scripted doorkey-5x5 0     # step through one episode
uv run harness report                          # re-score the latest run
```

### LLM agents

LLM agents are named `<strategy>[:<model>]` — `react`, `zero-shot:claude-haiku-4-5`.
They need `ANTHROPIC_API_KEY`; baselines don't. Always cap spend:

```bash
export ANTHROPIC_API_KEY=...
uv run harness run --agents scripted,zero-shot,react --tasks doorkey-5x5 \
    --seeds 5 --max-steps 60 --max-cost 2
uv run harness show react:claude-sonnet-5 doorkey-5x5 0   # reasoning, action, tokens, $ per step
```

`--max-steps` caps each episode, `--max-cost` stops the run between episodes once
it has spent that many dollars, and `--run-id` resumes a stopped run (after an
API outage too — finished episodes are saved as they complete).

| Strategy | How it acts |
|---|---|
| `zero-shot` | Forced `act` tool call, no visible reasoning — raw capability. |
| `react` | Reasons in plain text, then calls `act`; the reasoning is the step's thought. |

Each step is a fresh request (system prompt, the last `--memory` actions, the
current map), so cost per step stays flat however long an episode runs.

```
task           agent           n  success  return  steps
--------------------------------------------------------
doorkey-5x5    scripted       50     100%   0.968    9.0
doorkey-5x5    random         50      38%   0.176  149.3
doorkey-8x8    scripted       50     100%   0.976   17.2
doorkey-8x8    random         50      18%   0.066  451.2
```

## Design

| Piece | Where | Notes |
|---|---|---|
| Environment protocol | `envs/base.py` | Deterministic per seed. Observations carry a text rendering (what an LLM sees) and privileged structured state (only baselines may read it). |
| DoorKey | `envs/doorkey.py` | Dependency-free reimplementation of MiniGrid `DoorKey` semantics: same actions, reward `1 − 0.9·steps/max_steps`. Needs planning (key before door) and spatial reasoning (turn to face things). |
| Agents | `agents/` | `random` floor and `scripted` BFS ceiling; `llm.py` holds the prompting strategies, all behind the same protocol. |
| API client | `llmapi.py` | Standard-library Messages API client: retries on 429/5xx/overloaded with `retry-after`, token and cost accounting (`PRICES`). Injectable transport for offline tests. |
| Runner | `runner.py` | Runs the agent × task × seed matrix. Re-running a `--run-id` resumes; invalid actions are recorded, and five in a row end the episode. |
| Store | `store.py` | SQLite: one row per episode with token and cost totals, full trace as JSON. Older databases are migrated on open. |
| Scoring | `score.py` | Success rate, mean return, steps-to-goal, invalid-action rate, tokens and $ per episode. Bootstrapped CIs next. |

## Roadmap

1. ~~**LLM agent** — Anthropic client, text-observation prompt, tool-calling actions, token/cost/latency in traces, ReAct loop.~~
2. **Methodology** — multi-seed confidence intervals, more tasks, normalised scores.
3. **Dashboard** — leaderboard with CI bars and a step-through transcript viewer.
4. **Strategy study** — zero-shot vs few-shot vs ReAct vs Reflexion vs self-consistency, plus a write-up.

## License

MIT
