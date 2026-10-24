import type { Step } from "../types";
import { useTipHandlers } from "./Tooltip";

// The whole episode at a glance: one column per step. Output tokens rise as
// bars when the agent is an LLM (how much it "said"), invalid actions get a
// critical tick, and the cursor marks where playback is. Click to jump.
const W = 800;
const H = 54;
const BAR_H = 30;

export function Timeline({ trace, t, onSeek }: { trace: Step[]; t: number; onSeek: (t: number) => void }) {
  const tip = useTipHandlers();
  const n = trace.length;
  const col = W / Math.max(1, n + 1);
  const maxTok = Math.max(0, ...trace.map((s) => s.output_tokens ?? 0));
  const seek = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    onSeek(Math.max(0, Math.min(n, Math.floor(((e.clientX - r.left) / r.width) * (n + 1)))));
  };
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      preserveAspectRatio="none"
      style={{ display: "block", height: H, cursor: "pointer", touchAction: "none" }}
      onPointerDown={seek}
      onPointerMove={(e) => e.buttons === 1 && seek(e)}
      aria-hidden="true"
    >
      <line x1={0} x2={W} y1={BAR_H} y2={BAR_H} stroke="var(--axis)" />
      {trace.map((s) => {
        const x = s.t * col;
        const h = maxTok ? ((s.output_tokens ?? 0) / maxTok) * (BAR_H - 4) : 0;
        return (
          <g
            key={s.t}
            {...tip(
              <>
                <div className="tt-value">
                  step {s.t}: {s.action}
                  {s.invalid ? " (invalid)" : ""}
                </div>
                {s.output_tokens !== null && <div>{s.output_tokens} output tokens</div>}
              </>,
            )}
          >
            <rect x={x} y={0} width={col} height={H} fill="transparent" />
            {h > 0 && <rect x={x + col * 0.15} y={BAR_H - h} width={Math.max(1, col * 0.7)} height={h} fill="var(--accent)" opacity={0.55} />}
            {s.invalid && <rect x={x} y={BAR_H + 4} width={Math.max(2, col * 0.8)} height={8} rx={1} fill="var(--critical)" />}
          </g>
        );
      })}
      <rect x={t * col} y={0} width={Math.max(2, col)} height={H} fill="var(--ink)" opacity={0.18} />
      <rect x={t * col} y={BAR_H + 16} width={Math.max(2, col)} height={6} fill="var(--ink)" />
    </svg>
  );
}
