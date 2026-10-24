import type { Scale } from "../scale";
import type { Interval } from "../types";

// One estimate with its 95% interval on a shared axis: a 2px whisker and an
// 8px dot with a surface ring. Reference hairlines mark random (0) and
// scripted (1), so overlap between rows reads straight down the column.
export const CI_WIDTH = 240;
export const CI_HEIGHT = 22;

export function CIBar({ x, value, ci }: { x: Scale; value: number; ci: Interval }) {
  const y = CI_HEIGHT / 2;
  return (
    <svg width={CI_WIDTH} height={CI_HEIGHT} aria-hidden="true" style={{ display: "block" }}>
      <ReferenceLines x={x} height={CI_HEIGHT} />
      <line
        x1={x(ci[0])}
        x2={x(ci[1])}
        y1={y}
        y2={y}
        stroke="var(--accent)"
        strokeWidth={2}
        strokeLinecap="round"
      />
      <circle cx={x(value)} cy={y} r={5} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
    </svg>
  );
}

export function ReferenceLines({ x, height, top = 0 }: { x: Scale; height: number; top?: number }) {
  return (
    <g>
      {[0, 1].map((v) => (
        <line key={v} x1={x(v)} x2={x(v)} y1={top} y2={top + height} stroke="var(--axis)" strokeWidth={1} />
      ))}
    </g>
  );
}

/** Axis labels under the CI column: random at 0, scripted at 1. */
export function CIAxis({ x }: { x: Scale }) {
  return (
    <svg width={CI_WIDTH} height={14} aria-hidden="true" style={{ display: "block" }}>
      <text x={x(0)} y={11} textAnchor="start">
        0 random
      </text>
      <text x={x(1)} y={11} textAnchor="end">
        scripted 1
      </text>
    </svg>
  );
}
