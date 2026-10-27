# agent-eval-harness

**Which prompting strategy actually makes an LLM agent smarter — and how would you know?**

A small, honest harness for measuring LLM agents: run agents across a suite of
tasks and seeds, record every step (observation, reasoning, action, reward,
tokens, cost), score them with proper statistics, and explore the results.

> Status: **milestone 3 — dashboard.** A leaderboard with confidence intervals
> and paired "ahead / tied" verdicts, per-task small multiples, cost against
> score, and a step-through transcript viewer that redraws the grid the agent
> saw. Prompt-strategy study next.

## First results

[**Live dashboard →**](https://zdenekhynek.github.io/agent-eval-harness/) · run `sample-v1`:
3 gridworld tasks (5×5) × 10 held-out seeds, 40-step cap, $3.33 in total.

| Agent | Normalised score, 95% CI | Success | $/episode |
|---|---|---|---|
| react+think · Sonnet 5 | 1.00 [0.99, 1.00] | 100% | 0.026 |
| react · Sonnet 5 | 0.99 [0.98, 1.00] | 100% | 0.024 |
| zero-shot · Sonnet 5 | 0.99 [0.98, 1.00] | 100% | 0.019 |
| **react · Haiku 4.5** | **0.96 [0.93, 0.98]** | 100% | 0.023 |
| **zero-shot · Haiku 4.5** | **0.66 [0.52, 0.80]** | 70% | 0.019 |

- **ReAct matters a lot for the smaller model.** On Haiku 4.5, reasoning before
  acting lifts the score by **+0.30 [+0.15, +0.45]** (paired on seeds), entirely on
  the tasks that need it. Zero-shot walked into lava 3 times in 10 LavaGap
  episodes, once on step 0 with `Directly ahead: '~'` in its observation, and
  ran out of steps on 4 of 10 DoorKeys. ReAct never failed.
- **Sonnet 5 saturates these tasks.** Every strategy solves every seed; the
  differences left are a step or two per episode, real but well under the
  0.05 minimum effect, so the leaderboard reports them as *marginal*. The
  harder LavaCrossing tasks separate strong models: in a first look, Sonnet 5
  zero-shot ran out of steps on `lavacrossing-n2` where ReAct solved it in 21.
- **Hidden thinking would have confounded the study.** Sonnet 5 thinks privately
  by default, so out of the box "ReAct" had a reasoning channel that forced-tool
  "zero-shot" did not. Thinking is now explicitly off unless an agent is named
  `+think`.

Small sample (n = 10 per task): read the intervals, not the point estimates.

Write-up (draft): [Measuring LLM agents](docs/measuring-llm-agents.md) — the results and the
three measurement mistakes it took to get them right.

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
They need `ANTHROPIC_API_KEY`; baselines don't. Put it in a `.env` file in the
repo root (gitignored; read by `harness`, ignored by tests, never overriding a
variable already set) rather than a shell profile. Always cap spend:

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
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
| `few-shot` | Zero-shot plus four worked examples (turning, key, door, refusing lava), generated from scripted episodes on seeds no run evaluates. |
| `react` | Reasons in plain text, then calls `act`; the reasoning is the step's thought. |
| `reflexion` | ReAct plus in-episode self-reflection: when a situation repeats (a loop), the agent first writes a lesson that stays in its prompt. One attempt per seed, like every strategy. |
| `self-consistency` | Three ReAct samples in parallel; the majority action wins. Costs about 3×. |
| `<strategy>+think` | The same strategy with adaptive extended thinking on, e.g. `react+think`. Without the suffix every strategy sends `thinking: disabled`, so prompting is the only variable. Needs a model with adaptive thinking (Sonnet 5, not Haiku 4.5). |

Each step is a fresh request (system prompt, the last `--memory` actions, the
current map), so cost per step stays flat however long an episode runs.

## Dashboard

```bash
uv sync --extra serve
npm --prefix web ci && npm --prefix web run build
uv run harness serve                 # http://127.0.0.1:8000, live over runs.db
```

- **Leaderboard** — agents ranked by normalised score with a 95% CI bar on one
  shared axis. Each rank says whether it is *ahead* of the next (a paired
  comparison on shared seeds puts the CI of the difference above zero) or
  *tied*, so close point estimates are never over-read.
- **By task** — the same forest plot per task, same axis and agent order.
- **Cost against score** — what each LLM agent spends per episode for what it scores.
- **Episodes** — one cell per seed, shaded by score, hollow when failed. Each
  opens a **transcript**: the grid redrawn from the text the agent saw,
  its action, reasoning, tokens, cost and latency, with play, scrub and
  arrow-key stepping, and a timeline marking invalid actions.

`harness serve` and `harness export --out site` produce the same JSON at
the same paths (`data/runs.json`, `data/runs/<run>/…`), so the built app
works against either — live locally, or as static files. `scripts/build_site.sh`
builds the static site; the *Dashboard* workflow deploys it to GitHub Pages,
using `sample/runs.db` when one is committed and a fresh baseline run otherwise.

For frontend work, run `harness serve` and `npm --prefix web run dev`; Vite
proxies `data/` to it.

## Tasks

Dependency-free reimplementations of MiniGrid tasks, each isolating a skill:

| Family | Sizes | Tests | Notes |
|---|---|---|---|
| `empty` | 5, 6, 8 | spatial navigation | `Empty-Random`: random start pose, goal in the corner. |
| `lavagap` | 5, 6, 7 | reading the map before moving | `LavaGapS<N>`: stepping on lava ends the episode with reward 0. Few distinct layouts at small sizes. |
| `doorkey` | 5, 6, 8 | multi-step planning | Key, then door, then goal. |
| `lavacrossing-n1…n3` | 9 | route-finding past hazards | `LavaCrossingS9N<k>`: 1–3 lava rivers, one opening each, along a random monotone path. |

Iterate on prompts with the default seeds; report final numbers on held-out
seeds (`--seed-start 1000`) so a prompt tuned on seeds 0–19 is not graded on them.

## Methodology

| Question | How the harness answers it |
|---|---|
| How sure are we about a success rate? | Wilson 95% interval — honest at 0/n and n/n, where small LLM runs land. |
| How do tasks with different reward scales combine? | Normalised score: 0 = the random agent's mean return, 1 = the scripted solver's, measured on 100 fixed reference seeds (cached per harness version), never on the run being scored. |
| What is an agent's overall score? | Mean of per-task normalised scores with a bootstrap stratified by task, so tasks weigh equally. An agent missing a task is left out rather than scored on an easier subset. |
| Is agent A really better than B? | `harness compare A B`: paired bootstrap on per-seed differences (both agents saw the same layouts), per task and pooled. A win needs the 95% CI to exclude 0 **and** a difference of at least 0.05 (`--min-effect`); a real but smaller difference is reported as *marginal*, because near-deterministic agents make even one step on one seed "significant". |
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
| Views | `views.py` | The dashboard's JSON documents: leaderboard scopes with paired verdicts, per-episode traces with grid frames parsed from the observation text. `export` writes them to disk. |
| Server | `server.py` | FastAPI app (optional `serve` extra) serving the same documents live, plus the built web app. |
| Web | `web/` | React + TypeScript + Vite, TanStack Query, hand-built SVG charts; hash routing so deep links work on static hosting. |

## Roadmap

1. ~~**LLM agent** — Anthropic client, text-observation prompt, tool-calling actions, token/cost/latency in traces, ReAct loop.~~
2. ~~**Methodology** — multi-seed confidence intervals, more tasks, normalised scores.~~
3. ~~**Dashboard** — leaderboard with CI bars and a step-through transcript viewer.~~
4. **Strategy study** — zero-shot vs few-shot vs ReAct vs Reflexion vs self-consistency, a text-adventure env and hand-written puzzles, plus a write-up.

## License

MIT
