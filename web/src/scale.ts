export type Scale = ((v: number) => number) & { domain: [number, number]; ticks: number[] };

/** A linear scale from `domain` to `range`, with clean tick values inside the domain. */
export function linear(
  domain: [number, number],
  range: [number, number],
  tickCount = 5,
): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  const fn = ((v: number) => r0 + ((v - d0) / span) * (r1 - r0)) as Scale;
  fn.domain = domain;
  fn.ticks = niceTicks(d0, d1, tickCount);
  return fn;
}

export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = ([1, 2, 2.5, 5, 10].find((m) => m * mag >= raw) ?? 10) * mag;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) {
    ticks.push(Math.round(v / step) * step);
  }
  return ticks.map((v) => +v.toPrecision(12));
}

/** Domain covering every value and interval, padded, always including 0 and 1. */
export function normDomain(values: (number | null | undefined)[]): [number, number] {
  const finite = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const lo = Math.min(0, ...finite);
  const hi = Math.max(1, ...finite);
  const pad = (hi - lo) * 0.04;
  return [lo - pad, hi + pad];
}
