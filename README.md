# agent-eval-harness

**Which prompting strategy actually makes an LLM agent smarter — and how would you know?**

A small, honest harness for measuring LLM agents: run agents across a suite of
tasks and seeds, record every step (observation, reasoning, action, reward,
tokens, cost), score them with proper statistics, and explore the results.

> Status: **milestone 2 — methodology.** Three task families, 95% confidence
> intervals, scores normalised between a random floor and a scripted ceiling,
> and paired agent comparisons. Dashboard next.

## Quick start

```bash
uv sync --group dev
uv run harness run --tasks empty-6x6,lavagap-6x6,doorkey-6x6 --seeds 50
uv run harness report                          # re-score the latest run
uv run harness compare scripted random         # is the difference real?
uv run harness show scripted doorkey-6x6 0     # step through one episode
```

```
task        agent       n  success [95% CI]      norm [95% CI]  return  steps
------------------------------------------------------------------------------
doorkey-6x6 scripted   50   100% [93%,100%]   1.00 [1.00,1.00]   0.970   11.8
doorkey-6x6 random     50     24% [14%,37%]  0.02 [-0.05,0.10]   0.119  200.9
empty-6x6   scripted   50   100% [93%,100%]   0.99 [0.99,1.00]   0.969    5.0
empty-6x6   random     50     70% [56%,81%] -0.07 [-0.25,0.12]   0.417   64.7
lavagap-6x6 scripted   50   100% [93%,100%]   1.00 [1.00,1.00]   0.950    8.1
lavagap-6x6 random     50       4% [1%,13%] -0.01 [-0.04,0.03]   0.028   50.0

overall: mean normalised score across tasks (0 = random, 1 = scripted)
agent    tasks     n      norm [95% CI]
---------------------------------------
scripted     3   150   1.00 [1.00,1.00]
random       3   150 -0.02 [-0.09,0.05]
```

Random "succeeds" 70% of the time on `empty-6x6` yet scores zero: stumbling onto
the goal in 65 steps is what the floor looks like. That is why the headline
metric is the normalised score, not the success rate.

### LLM agents

LLM agents are named `<strategy>[:<model>]` — `react`, `zero-shot:claude-haiku-4-5`.
They need `ANTHROPIC_API_KEY`; baselines don't. Always cap spend:

```bash
export ANTHROPIC_API_KEY=...
uv run harness run --agents scripted,zero-shot,react \
    --tasks empty-5x5,lavagap-5x5,doorkey-5x5 --seeds 5 --max-steps 60 --max-cost 2
uv run harness compare react:claude-sonnet-5 zero-shot:claude-sonnet-5
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

## Tasks

Dependency-free reimplementations of MiniGrid tasks, each isolating a skill:

| Family | Sizes | Tests | Notes |
|---|---|---|---|
| `empty` | 5, 6, 8 | spatial navigation | `Empty-Random`: random start pose, goal in the corner. |
| `lavagap` | 5, 6, 7 | reading the map before moving | `LavaGapS<N>`: stepping on lava ends the episode with reward 0. Few distinct layouts at small sizes. |
| `doorkey` | 5, 6, 8 | multi-step planning | Key, then door, then goal. |

Iterate on prompts with the default seeds; report final numbers on held-out
seeds (`--seed-start 1000`) so a prompt tuned on seeds 0–19 is not graded on them.

## Methodology

| Question | How the harness answers it |
|---|---|
| How sure are we about a success rate? | Wilson 95% interval — honest at 0/n and n/n, where small LLM runs land. |
| How do tasks with different reward scales combine? | Normalised score: 0 = the random agent's mean return, 1 = the scripted solver's, measured on 100 fixed reference seeds (cached per harness version), never on the run being scored. |
| What is an agent's overall score? | Mean of per-task normalised scores with a bootstrap stratified by task, so tasks weigh equally. An agent missing a task is left out rather than scored on an easier subset. |
| Is agent A really better than B? | `harness compare A B`: paired bootstrap on per-seed differences (both agents saw the same layouts), per task and pooled. The verdict stays "no clear difference" unless the 95% CI excludes 0. |
| Did the prompt change halfway through a run? | Each LLM agent has a fingerprint (prompts, model, memory, temperature, max tokens). Resuming a run with a different fingerprint or `--max-steps` is refused. |

Bootstraps use a fixed RNG seed, so a report is reproducible.

## Design

| Piece | Where | Notes |
|---|---|---|
| Environment protocol | `envs/base.py` | Deterministic per seed. Observations carry a text rendering (what an LLM sees) and privileged structured state (only baselines may read it). |
| Gridworlds | `envs/grid.py` | Shared MiniGrid mechanics: turn-then-move, interact with the cell ahead, reward `1 − 0.9·steps/max_steps`. `doorkey.py` and `navigation.py` only lay out their grids. |
| Agents | `agents/` | `random` floor and `scripted` ceiling (BFS over the glyph grid, avoiding lava); `llm.py` holds the prompting strategies, all behind the same protocol. |
| API client | `llmapi.py` | Standard-library Messages API client: retries on 429/5xx/overloaded with `retry-after`, token and cost accounting (`PRICES`). Injectable transport for offline tests. |
| Runner | `runner.py` | Runs the agent × task × seed matrix. Re-running a `--run-id` resumes, guarded by agent fingerprints; invalid actions are recorded, and five in a row end the episode. Computes normalisation references. |
| Store | `store.py` | SQLite: one row per episode with token and cost totals and the full trace as JSON; run configs; cached references. Older databases are migrated on open. |
| Scoring | `score.py` | Wilson and bootstrap CIs, normalised and overall scores, paired comparisons, plus steps-to-goal, invalid-action rate, tokens and $ per episode. |

## Roadmap

1. ~~**LLM agent** — Anthropic client, text-observation prompt, tool-calling actions, token/cost/latency in traces, ReAct loop.~~
2. ~~**Methodology** — multi-seed confidence intervals, more tasks, normalised scores.~~
3. **Dashboard** — leaderboard with CI bars and a step-through transcript viewer.
4. **Strategy study** — zero-shot vs few-shot vs ReAct vs Reflexion vs self-consistency, a text-adventure env and hand-written puzzles, plus a write-up.

## License

MIT
