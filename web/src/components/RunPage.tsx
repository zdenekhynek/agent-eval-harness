import { useState } from "react";
import { useLeaderboard, useRuns } from "../api";
import { ci2, compact, isBaseline, num2, usd } from "../format";
import { href } from "../route";
import type { Leaderboard, RunSummary } from "../types";
import { AgentName } from "./AgentName";
import { CostScatter } from "./CostScatter";
import { EpisodeStrips } from "./EpisodeStrips";
import { LeaderboardTable } from "./LeaderboardTable";
import { TaskForests } from "./TaskForests";

export function RunPage({ run, scope }: { run?: string; scope: string }) {
  const runs = useRuns();
  if (runs.isPending) return <p className="note">Loading runs…</p>;
  if (runs.isError) return <ErrorCard error={runs.error} />;
  const list = runs.data.runs;
  if (!list.length) {
    return (
      <div className="card">
        <h2>No runs yet</h2>
        <p className="sub">
          Start one with <code>uv run harness run</code>, then reload.
        </p>
      </div>
    );
  }
  const current = list.find((r) => r.id === run) ?? list[0]!;
  return <RunView runs={list} run={current} scope={scope} />;
}

function RunView({ runs, run, scope }: { runs: RunSummary[]; run: RunSummary; scope: string }) {
  const board = useLeaderboard(run.id);
  const [hot, setHot] = useState<string | null>(null);

  const go = (next: { run?: string; scope?: string }) => {
    window.location.hash = href({ page: "run", run: next.run ?? run.id, scope: next.scope ?? scope });
  };

  return (
    <>
      <div className="filters">
        <label>
          Run
          <select value={run.id} onChange={(e) => go({ run: e.target.value, scope: "all" })}>
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.id} · {r.episodes} episodes
              </option>
            ))}
          </select>
        </label>
        {board.data && (
          <div className="segmented" role="group" aria-label="Task">
            {board.data.scopes.map((s) => (
              <button key={s.id} aria-pressed={s.id === scope} onClick={() => go({ scope: s.id })}>
                {s.label}
              </button>
            ))}
          </div>
        )}
      </div>
      {board.isError && <ErrorCard error={board.error} />}
      {board.data && (
        <div style={{ opacity: board.isPlaceholderData ? 0.5 : 1, transition: "opacity 150ms" }}>
          <Board board={board.data} run={run} scopeId={scope} hot={hot} setHot={setHot} />
        </div>
      )}
    </>
  );
}

function Board({
  board,
  run,
  scopeId,
  hot,
  setHot,
}: {
  board: Leaderboard;
  run: RunSummary;
  scopeId: string;
  hot: string | null;
  setHot: (a: string | null) => void;
}) {
  const scope = board.scopes.find((s) => s.id === scopeId) ?? board.scopes[0]!;
  const overallScope = board.scopes[0]!;
  const taskScopes = board.scopes.slice(1);
  const order = scope.rows.map((r) => r.agent);
  for (const a of board.agents) if (!order.includes(a)) order.push(a);
  const best = scope.rows.find((r) => !isBaseline(r.agent));
  const tasks = scope.id === "all" ? board.tasks : [scope.id];
  const episodes = board.episodes.filter((e) => tasks.includes(e.task));
  const cost = episodes.reduce((sum, e) => sum + (e.cost_usd ?? 0), 0);

  return (
    <>
      <div className="kpis">
        <div className="kpi">
          <div className="label">Best LLM agent</div>
          <div className="value">{best?.norm != null ? num2(best.norm) : "–"}</div>
          <div className="detail">
            {best ? (
              <>
                <AgentName agent={best.agent} /> {best.norm_ci && ci2(best.norm_ci)}
              </>
            ) : (
              "baselines only in this run"
            )}
          </div>
        </div>
        <div className="kpi">
          <div className="label">Episodes</div>
          <div className="value">{compact(episodes.length)}</div>
          <div className="detail">
            {order.length} agents × {tasks.length} {tasks.length === 1 ? "task" : "tasks"}
          </div>
        </div>
        <div className="kpi">
          <div className="label">Spend</div>
          <div className="value">{cost ? usd(cost) : "$0"}</div>
          <div className="detail">{cost ? "API cost of these episodes" : "no priced agents"}</div>
        </div>
        <div className="kpi">
          <div className="label">Run</div>
          <div className="value" style={{ fontSize: 16, wordBreak: "break-all" }}>
            {run.id}
          </div>
          <div className="detail">{new Date(run.created_at).toLocaleString()}</div>
        </div>
      </div>

      <section className="card">
        <h2>Leaderboard · {scope.label}</h2>
        <p className="sub">
          Ranked by normalised score: 0 is the random agent, 1 the scripted solver.
          {scope.id === "all" && " Tasks weigh equally."} “Ahead” means a paired comparison on
          shared seeds puts the 95% CI of the difference above zero; otherwise the two are tied.
        </p>
        <LeaderboardTable scope={scope} hot={hot} onHover={setHot} />
        {scope.excluded.length > 0 && (
          <p className="note">
            Not ranked across all tasks (skipped one or more): {scope.excluded.join(", ")}
          </p>
        )}
      </section>

      {scope.id === "all" && taskScopes.length > 1 && (
        <section className="card">
          <h2>By task</h2>
          <p className="sub">Normalised score with 95% CI, same axis and agent order in every panel.</p>
          <TaskForests scopes={taskScopes} order={order} hot={hot} />
        </section>
      )}

      <section className="card">
        <h2>Cost against score</h2>
        <p className="sub">Mean API cost per episode against normalised score, priced agents only.</p>
        <CostScatter rows={(scope.id === "all" ? overallScope : scope).rows} hot={hot} />
      </section>

      <section className="card">
        <h2>Episodes</h2>
        <p className="sub">One cell per seed. Open any cell to step through its transcript.</p>
        <EpisodeStrips run={run.id} tasks={tasks} order={order} episodes={episodes} />
      </section>
    </>
  );
}

export function ErrorCard({ error }: { error: Error }) {
  return (
    <div className="card error" role="alert">
      <h2>Couldn’t load data</h2>
      <p className="sub">{error.message}</p>
      <p className="note">
        Serving locally? Run <code>uv run harness serve</code>, or export with{" "}
        <code>uv run harness export</code>.
      </p>
    </div>
  );
}
