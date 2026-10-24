import { useEffect, useState } from "react";
import { EpisodePage } from "./components/EpisodePage";
import { RunPage } from "./components/RunPage";
import { useRoute } from "./route";

type Theme = "light" | "dark" | null;

function readTheme(): Theme {
  try {
    const v = localStorage.getItem("theme");
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

export function App() {
  const route = useRoute();
  const [theme, setTheme] = useState<Theme>(readTheme);

  useEffect(() => {
    const root = document.documentElement;
    if (theme) root.dataset.theme = theme;
    else delete root.dataset.theme;
    try {
      if (theme) localStorage.setItem("theme", theme);
      else localStorage.removeItem("theme");
    } catch {
      // storage blocked: the choice lasts for this page view only
    }
  }, [theme]);

  const dark =
    theme === "dark" || (theme === null && window.matchMedia("(prefers-color-scheme: dark)").matches);

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#/">
          Agent eval harness <span>· which prompting strategy makes an agent smarter?</span>
        </a>
        <a className="note" href="https://github.com/zdenekhynek/agent-eval-harness">
          GitHub
        </a>
        <button className="icon-btn" onClick={() => setTheme(dark ? "light" : "dark")} aria-label="Toggle dark mode">
          {dark ? "Light" : "Dark"}
        </button>
      </header>
      <main>
        {route.page === "episode" ? (
          <EpisodePage run={route.run} agent={route.agent} task={route.task} seed={route.seed} t={route.t} />
        ) : (
          <RunPage run={route.run} scope={route.scope} />
        )}
      </main>
    </div>
  );
}
