import { ci2, num2, splitAgent, usd } from "../format";
import { linear } from "../scale";
import type { Row } from "../types";
import { useTipHandlers } from "./Tooltip";

// Cost/performance: what each priced agent spends per episode against what it
// scores. One hue with direct labels; the vertical whisker is the score's CI.
const W = 680;
const H = 300;
const M = { top: 16, right: 150, bottom: 36, left: 44 };

export function CostScatter({ rows, hot }: { rows: Row[]; hot: string | null }) {
  const tip = useTipHandlers();
  const priced = rows.filter(
    (r): r is Row & { cost_usd: number; norm: number } => r.cost_usd !== null && r.norm !== null,
  );
  if (priced.length === 0) {
    return <p className="note">No priced (LLM) agents in this scope, so there is no cost to plot.</p>;
  }
  const maxCost = Math.max(...priced.map((r) => r.cost_usd)) * 1.1 || 1;
  const lows = priced.map((r) => r.norm_ci?.[0] ?? r.norm);
  const highs = priced.map((r) => r.norm_ci?.[1] ?? r.norm);
  const x = linear([0, maxCost], [M.left, W - M.right]);
  // Zoom to the data, but keep the scripted ceiling (1) in view for reference.
  const top = Math.max(1, ...highs);
  const bottom = Math.min(...lows);
  const pad = Math.max(0.02, (top - bottom) * 0.08);
  const y = linear([Math.floor((bottom - pad) * 20) / 20, top + pad / 2], [H - M.bottom, M.top], 4);
  const labels = placeLabels(priced.map((r) => ({ agent: r.agent, x: x(r.cost_usd), y: y(r.norm) })));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W, display: "block" }} role="img" aria-label="Normalised score against cost per episode" className={hot ? "dim" : undefined}>
      {y.ticks.map((t) => (
        <g key={t}>
          <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
          <text x={M.left - 8} y={y(t) + 4} textAnchor="end">
            {num2(t)}
          </text>
        </g>
      ))}
      {x.ticks.map((t) => (
        <text key={t} x={x(t)} y={H - M.bottom + 16} textAnchor="middle">
          {usd(t)}
        </text>
      ))}
      <line x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} stroke="var(--axis)" />
      <text x={M.left} y={H - 4}>
        cost per episode →
      </text>
      <text x={M.left - 36} y={M.top - 4}>
        score
      </text>
      {priced.map((r) => {
        const [strategy, model] = splitAgent(r.agent);
        const cx = x(r.cost_usd);
        const cy = y(r.norm);
        return (
          <g
            key={r.agent}
            className={`mark${hot === r.agent ? " hot" : ""}`}
            tabIndex={0}
            {...tip(
              <>
                <div className="tt-value">
                  {num2(r.norm)} {r.norm_ci ? ci2(r.norm_ci) : ""} at {usd(r.cost_usd)}/ep
                </div>
                <div>{r.agent}</div>
              </>,
            )}
          >
            {r.norm_ci && (
              <line x1={cx} x2={cx} y1={y(r.norm_ci[0])} y2={y(r.norm_ci[1])} stroke="var(--accent)" strokeWidth={2} strokeLinecap="round" />
            )}
            <circle cx={cx} cy={cy} r={12} fill="transparent" />
            <circle cx={cx} cy={cy} r={5} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
            {labels.has(r.agent) && (
              <text {...labels.get(r.agent)} style={{ fill: "var(--ink-2)" }}>
                {strategy}
                {model ? ` · ${model}` : ""}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

interface Placed {
  x: number;
  y: number;
  textAnchor: "start" | "end" | "middle";
}

type Box = { x0: number; x1: number; y0: number; y1: number };
const overlaps = (a: Box, b: Box) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;

/**
 * Direct labels that never move off their dot: try right of the dot, then
 * left, below and above, and keep the first spot clear of every dot and every
 * earlier label. A label with no clear spot is left off - the tooltip and the
 * leaderboard table still carry it - rather than stacked on a neighbour.
 */
export function placeLabels(points: { agent: string; x: number; y: number }[]): Map<string, Placed> {
  const CHAR = 6.2;
  const DOT = 7;
  const taken: Box[] = points.map((p) => ({ x0: p.x - DOT, x1: p.x + DOT, y0: p.y - DOT, y1: p.y + DOT }));
  const placed = new Map<string, Placed>();
  for (const p of [...points].sort((a, b) => a.y - b.y)) {
    const w = p.agent.replace(":", " · ").length * CHAR;
    const options: [Placed, number][] = [
      [{ x: p.x + 10, y: p.y + 4, textAnchor: "start" }, p.x + 10],
      [{ x: p.x - 10, y: p.y + 4, textAnchor: "end" }, p.x - 10 - w],
      [{ x: p.x, y: p.y + 20, textAnchor: "middle" }, p.x - w / 2],
      [{ x: p.x, y: p.y - 12, textAnchor: "middle" }, p.x - w / 2],
    ];
    for (const [opt, x0] of options) {
      const box = { x0, x1: x0 + w, y0: opt.y - 10, y1: opt.y + 3 };
      if (!taken.some((b) => overlaps(b, box))) {
        placed.set(p.agent, opt);
        taken.push(box);
        break;
      }
    }
  }
  return placed;
}
