import { ci2, num2, pct, splitAgent } from "../format";
import { linear, normDomain } from "../scale";
import type { Scope } from "../types";
import { ReferenceLines } from "./CIBar";
import { useTipHandlers } from "./Tooltip";

// Small multiples: one forest plot per task, agents in the overall ranking
// order and one shared x-axis, so a row can be compared across tasks.
const W = 340;
const LABEL = 128;
const ROW = 22;
const PAD_X = 10;

export function TaskForests({
  scopes,
  order,
  hot,
}: {
  scopes: Scope[];
  order: string[];
  hot: string | null;
}) {
  const tip = useTipHandlers();
  const all = scopes.flatMap((s) => s.rows.flatMap((r) => [r.norm, ...(r.norm_ci ?? [])]));
  const x = linear(normDomain(all), [LABEL + PAD_X, W - PAD_X]);
  const height = order.length * ROW + 22;

  return (
    <div className={`chart-grid${hot ? " dim" : ""}`}>
      {scopes.map((scope) => {
        const byAgent = new Map(scope.rows.map((r) => [r.agent, r]));
        return (
          <figure key={scope.id} style={{ margin: 0 }}>
            <figcaption className="chart-title">{scope.label}</figcaption>
            <svg viewBox={`0 0 ${W} ${height}`} width="100%" role="img" aria-label={`Normalised score by agent on ${scope.label}`}>
              <ReferenceLines x={x} height={order.length * ROW} top={2} />
              {order.map((agent, i) => {
                const row = byAgent.get(agent);
                const y = i * ROW + ROW / 2 + 2;
                const [strategy, model] = splitAgent(agent);
                const label = model ? `${strategy} · ${model}` : strategy;
                return (
                  <g key={agent} className={`mark${hot === agent ? " hot" : ""}`}>
                    <text x={0} y={y + 4} style={{ fill: "var(--ink-2)" }}>
                      {label.length > 22 ? `${label.slice(0, 21)}…` : label}
                    </text>
                    {row && row.norm !== null && row.norm_ci ? (
                      <g
                        tabIndex={0}
                        {...tip(
                          <>
                            <div className="tt-value">
                              {num2(row.norm)} {ci2(row.norm_ci)}
                            </div>
                            <div>
                              {agent} on {scope.label}
                            </div>
                            <div>
                              success {pct(row.success)} · n = {row.n}
                            </div>
                          </>,
                        )}
                      >
                        <rect x={LABEL} y={y - ROW / 2} width={W - LABEL} height={ROW} fill="transparent" />
                        <line
                          x1={x(row.norm_ci[0])}
                          x2={x(row.norm_ci[1])}
                          y1={y}
                          y2={y}
                          stroke="var(--accent)"
                          strokeWidth={2}
                          strokeLinecap="round"
                        />
                        <circle cx={x(row.norm)} cy={y} r={4.5} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
                      </g>
                    ) : (
                      <text x={LABEL + PAD_X} y={y + 4}>
                        not run
                      </text>
                    )}
                  </g>
                );
              })}
              <text x={x(0)} y={height - 4} textAnchor="middle">
                0
              </text>
              <text x={x(1)} y={height - 4} textAnchor="middle">
                1
              </text>
            </svg>
          </figure>
        );
      })}
    </div>
  );
}
