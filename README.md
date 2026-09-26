# agent-eval-harness

**Which prompting strategy actually makes an LLM agent smarter — and how would you know?**

A small, honest harness for measuring LLM agents: run agents across a suite of
tasks and seeds, record every step (observation, reasoning, action, reward,
tokens, cost), score them with proper statistics, and explore the results.

> Status: **milestone 0 — skeleton.** Environments, baselines, runner, storage,
> CLI. No LLM agents yet; those arrive next.

## Quick start

```bash
uv sync --group dev
uv run harness run --tasks doorkey-5x5,doorkey-8x8 --seeds 50
uv run harness show scripted doorkey-5x5 0     # step through one episode
uv run harness report                          # re-score the latest run
```

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
| Agents | `agents/` | `random` floor and `scripted` BFS ceiling. LLM prompting strategies (zero-shot, ReAct, Reflexion, …) plug into the same protocol. |
| Runner | `runner.py` | Runs the agent × task × seed matrix. Re-running a `--run-id` resumes; invalid actions are recorded, and five in a row end the episode. |
| Store | `store.py` | SQLite: one row per episode, full trace as JSON. |
| Scoring | `score.py` | Success rate, mean return, steps-to-goal. Bootstrapped CIs next. |

## Roadmap

1. **LLM agent** — Anthropic client, text-observation prompt, tool-calling actions, token/cost/latency in traces, ReAct loop.
2. **Methodology** — multi-seed confidence intervals, more tasks, normalised scores.
3. **Dashboard** — leaderboard with CI bars and a step-through transcript viewer.
4. **Strategy study** — zero-shot vs few-shot vs ReAct vs Reflexion vs self-consistency, plus a write-up.

## License

MIT
