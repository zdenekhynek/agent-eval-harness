import { describe, expect, it } from "vitest";
import { agentSlug, episodePath } from "./api";
import { placeLabels } from "./components/CostScatter";
import { seqStep } from "./components/EpisodeStrips";
import { compact, num2, signed2, splitAgent, usd } from "./format";
import { href, parseRoute } from "./route";
import { linear, niceTicks, normDomain } from "./scale";

describe("routes", () => {
  it("round-trips episode routes, including agent names with colons", () => {
    const route = { page: "episode", run: "run-1", agent: agentSlug("react:claude-sonnet-5"), task: "doorkey-5x5", seed: 7, t: 3 } as const;
    expect(parseRoute(href(route))).toEqual(route);
  });

  it("defaults to the latest run, all tasks", () => {
    expect(parseRoute("")).toEqual({ page: "run", run: undefined, scope: "all" });
    expect(parseRoute("#/run/abc?scope=empty-5x5")).toEqual({ page: "run", run: "abc", scope: "empty-5x5" });
  });

  it("falls back to the run page for malformed episode links", () => {
    expect(parseRoute("#/run/abc/ep/x/y/notaseed").page).toBe("run");
  });

  it("builds the same episode path as the Python exporter", () => {
    expect(episodePath("r", "react:claude-sonnet-5", "empty-5x5", 2)).toBe(
      "data/runs/r/episodes/react~claude-sonnet-5/empty-5x5/2.json",
    );
  });
});

describe("format", () => {
  it("formats numbers for tables", () => {
    expect(num2(-0.001)).toBe("0.00");
    expect(signed2(-0.25)).toBe("−0.25");
    expect(signed2(0.004)).toBe("+0.004");
    expect(signed2(0)).toBe("+0.00");
    expect(compact(12_345)).toBe("12.3K");
    expect(usd(0.0012)).toBe("$0.0012");
    expect(usd(null)).toBe("–");
    expect(splitAgent("zero-shot:claude-haiku-4-5")).toEqual(["zero-shot", "claude-haiku-4-5"]);
    expect(splitAgent("random")).toEqual(["random", null]);
  });
});

describe("scales", () => {
  it("picks clean ticks", () => {
    expect(niceTicks(0, 1, 5)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    expect(niceTicks(0, 0.088, 4)).toEqual([0, 0.025, 0.05, 0.075]);
  });

  it("always shows the random and scripted anchors", () => {
    const [lo, hi] = normDomain([0.4, 0.6]);
    expect(lo).toBeLessThan(0);
    expect(hi).toBeGreaterThan(1);
    expect(linear([0, 1], [0, 100])(0.25)).toBe(25);
  });

  it("bins normalised scores onto the ramp", () => {
    expect(seqStep(-0.3)).toBe(0);
    expect(seqStep(0.5)).toBe(3);
    expect(seqStep(1)).toBe(6);
    expect(seqStep(null)).toBe(0);
  });
});

describe("scatter labels", () => {
  it("never places a label over a dot or another label, dropping it instead", () => {
    const placed = placeLabels([
      { agent: "react:claude-sonnet-5", x: 300, y: 50 },
      { agent: "zero-shot:claude-sonnet-5", x: 305, y: 52 },
      { agent: "react:claude-haiku-4-5", x: 100, y: 200 },
    ]);
    expect(placed.has("react:claude-haiku-4-5")).toBe(true);
    expect(placed.size).toBeGreaterThanOrEqual(2);
  });
});
