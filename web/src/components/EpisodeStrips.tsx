import { agentSlug } from "../api";
import { num2, outcome, usd } from "../format";
import { href } from "../route";
import type { EpisodeRow } from "../types";
import { AgentName } from "./AgentName";
import { useTipHandlers } from "./Tooltip";

// Every episode as a cell: agents down, seeds across, one block per task.
// Solved episodes are shaded on the one-hue ramp by normalised score (darker =
// closer to the scripted solver); failures are hollow with a glyph, so outcome
// never depends on colour alone. Each cell opens that episode's transcript.
const STEPS = 7;

export function seqStep(norm: number | null): number {
  const v = Math.min(1, Math.max(0, norm ?? 0));
  return Math.min(STEPS - 1, Math.floor(v * STEPS));
}

export function EpisodeStrips({
  run,
  tasks,
  order,
  episodes,
}: {
  run: string;
  tasks: string[];
  order: string[];
  episodes: EpisodeRow[];
}) {
  const tip = useTipHandlers();
  return (
    <>
      <div className="strips">
        {tasks.map((task) => (
          <div key={task}>
            <div className="chart-title">{task}</div>
            {order.map((agent) => {
              const cells = episodes
                .filter((e) => e.task === task && e.agent === agent)
                .sort((a, b) => a.seed - b.seed);
              if (!cells.length) return null;
              return (
                <div className="strip-row" key={agent}>
                  <div className="strip-label">
                    <AgentName agent={agent} />
                  </div>
                  <div className="cells">
                    {cells.map((e) => (
                      <a
                        key={e.seed}
                        className={`cell${e.success ? "" : " fail"}`}
                        style={e.success ? { background: `var(--seq-${seqStep(e.norm)})` } : undefined}
                        href={href({ page: "episode", run, agent: agentSlug(agent), task, seed: e.seed, t: 0 })}
                        aria-label={`${agent}, ${task}, seed ${e.seed}: ${outcome(e)} in ${e.steps} steps`}
                        {...tip(
                          <>
                            <div className="tt-value">
                              {outcome(e)} in {e.steps} steps
                            </div>
                            <div>
                              seed {e.seed} · score {e.norm !== null ? num2(e.norm) : "–"} · return {e.return.toFixed(3)}
                            </div>
                            {e.invalid > 0 && <div>{e.invalid} invalid actions</div>}
                            {e.cost_usd !== null && <div>{usd(e.cost_usd)}</div>}
                          </>,
                        )}
                      >
                        {e.success ? "" : e.truncated ? "⋯" : "×"}
                      </a>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="legend">
        <span>
          solved, score 0
          <span className="ramp" aria-hidden="true">
            {Array.from({ length: STEPS }, (_, i) => (
              <i key={i} style={{ background: `var(--seq-${i})` }} />
            ))}
          </span>
          1
        </span>
        <span>
          <span className="cell fail" style={{ display: "inline-grid", marginRight: 6 }}>
            ×
          </span>
          failed
        </span>
        <span>
          <span className="cell fail" style={{ display: "inline-grid", marginRight: 6 }}>
            ⋯
          </span>
          ran out of steps
        </span>
      </div>
    </>
  );
}
