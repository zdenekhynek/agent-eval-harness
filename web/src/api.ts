import { useQuery } from "@tanstack/react-query";
import type { EpisodeDoc, Leaderboard, RunSummary } from "./types";

// Every path is relative to the page, so the build works under any base path
// and against either `harness serve` or a static `harness export`.
async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${path}`);
  return (await res.json()) as T;
}

export const agentSlug = (agent: string) => agent.replaceAll(":", "~");

export const episodePath = (run: string, agent: string, task: string, seed: number) =>
  `data/runs/${encodeURIComponent(run)}/episodes/${encodeURIComponent(agentSlug(agent))}/` +
  `${encodeURIComponent(task)}/${seed}.json`;

export const useRuns = () =>
  useQuery({
    queryKey: ["runs"],
    queryFn: () => getJson<{ runs: RunSummary[] }>("data/runs.json"),
  });

export const useLeaderboard = (run: string | undefined) =>
  useQuery({
    queryKey: ["leaderboard", run],
    queryFn: () =>
      getJson<Leaderboard>(`data/runs/${encodeURIComponent(run ?? "")}/leaderboard.json`),
    enabled: run !== undefined,
    placeholderData: (previous) => previous,
  });

export const useEpisode = (run: string, agent: string, task: string, seed: number) =>
  useQuery({
    queryKey: ["episode", run, agent, task, seed],
    queryFn: () => getJson<EpisodeDoc>(episodePath(run, agent, task, seed)),
    placeholderData: (previous) => previous,
  });
