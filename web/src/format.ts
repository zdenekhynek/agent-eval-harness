import type { Interval } from "./types";

export const pct = (v: number) => `${Math.round(v * 100)}%`;
export const num2 = (v: number) => (Math.abs(v) < 0.005 ? "0.00" : v.toFixed(2));
/** Signed difference; small non-zero values get a third decimal so "+0.00" never appears. */
export const signed2 = (v: number) => {
  const a = Math.abs(v);
  const digits = a > 0 && a < 0.01 ? 3 : 2;
  return (v >= 0 ? "+" : "−") + a.toFixed(digits);
};

export const signedCi = ([lo, hi]: Interval) => `[${signed2(lo)}, ${signed2(hi)}]`;
export const ci2 = ([lo, hi]: Interval) => `[${num2(lo)}, ${num2(hi)}]`;
export const ciPct = ([lo, hi]: Interval) => `[${pct(lo)}, ${pct(hi)}]`;

export function compact(v: number): string {
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (Math.abs(v) >= 1e4) return `${(v / 1e3).toFixed(1)}K`;
  return Math.round(v).toLocaleString("en");
}

export function usd(v: number | null | undefined): string {
  if (v === null || v === undefined) return "–";
  if (v === 0) return "$0";
  if (v < 0.01) return `$${v.toFixed(4)}`;
  if (v < 10) return `$${v.toFixed(3)}`;
  return `$${v.toFixed(2)}`;
}

/** Strategy and model from an agent name: "react:claude-sonnet-5" -> ["react", "claude-sonnet-5"]. */
export function splitAgent(agent: string): [string, string | null] {
  const i = agent.indexOf(":");
  return i < 0 ? [agent, null] : [agent.slice(0, i), agent.slice(i + 1)];
}

export const isBaseline = (agent: string) => splitAgent(agent)[1] === null;

export const outcome = (e: { success: boolean; truncated: boolean }) =>
  e.success ? "solved" : e.truncated ? "ran out of steps" : "failed";
