import { useEffect, useState } from "react";

// Hash routes, so deep links work on static hosting with no rewrite rules:
//   #/                                  latest run
//   #/run/<run>?scope=<task|all>        a run's leaderboard
//   #/run/<run>/ep/<agent>/<task>/<seed>?t=<step>
export type Route =
  | { page: "run"; run?: string; scope: string }
  | { page: "episode"; run: string; agent: string; task: string; seed: number; t: number };

export function parseRoute(hash: string): Route {
  const [path = "", query = ""] = hash.replace(/^#/, "").split("?");
  const params = new URLSearchParams(query);
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  if (parts[0] === "run" && parts[2] === "ep" && parts.length === 6) {
    const seed = Number(parts[5]);
    if (Number.isInteger(seed)) {
      return {
        page: "episode",
        run: parts[1]!,
        agent: parts[3]!,
        task: parts[4]!,
        seed,
        t: Math.max(0, Number(params.get("t")) || 0),
      };
    }
  }
  return {
    page: "run",
    run: parts[0] === "run" ? parts[1] : undefined,
    scope: params.get("scope") ?? "all",
  };
}

export function href(route: Route): string {
  const e = encodeURIComponent;
  if (route.page === "episode") {
    const q = route.t ? `?t=${route.t}` : "";
    return `#/run/${e(route.run)}/ep/${e(route.agent)}/${e(route.task)}/${route.seed}${q}`;
  }
  const q = route.scope !== "all" ? `?scope=${e(route.scope)}` : "";
  return route.run ? `#/run/${e(route.run)}${q}` : `#/${q}`;
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseRoute(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

/** Update the URL without adding a history entry (used while scrubbing). */
export function replaceRoute(route: Route): void {
  window.history.replaceState(null, "", href(route));
}
