import { useCallback, useEffect, useState } from "react";
import { useEpisode } from "../api";
import { compact, num2, outcome, usd } from "../format";
import { href, replaceRoute } from "../route";
import type { EpisodeDoc } from "../types";
import { AgentName } from "./AgentName";
import { BoardLegend, GridBoard } from "./GridBoard";
import { ErrorCard } from "./RunPage";
import { Timeline } from "./Timeline";

interface Props {
  run: string;
  agent: string; // slug form, "~" for ":"
  task: string;
  seed: number;
  t: number;
}

export function EpisodePage(props: Props) {
  const agent = props.agent.replaceAll("~", ":");
  const ep = useEpisode(props.run, agent, props.task, props.seed);
  if (ep.isPending) return <p className="note">Loading episode…</p>;
  if (ep.isError) return <ErrorCard error={ep.error} />;
  return <Player key={`${agent}/${props.task}/${props.seed}`} ep={ep.data} {...props} />;
}

const SPEEDS = [1, 4, 12];

const ms = (v: number) => (v < 1 ? "<1 ms" : `${Math.round(v).toLocaleString("en")} ms`);

function Player({ ep, run, agent: slug, task, seed, t: initial }: Props & { ep: EpisodeDoc }) {
  const n = ep.trace.length;
  const [t, setT] = useState(Math.min(initial, n));
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);

  const seek = useCallback(
    (next: number) => {
      const clamped = Math.max(0, Math.min(n, next));
      setT(clamped);
      replaceRoute({ page: "episode", run, agent: slug, task, seed, t: clamped });
    },
    [n, run, slug, task, seed],
  );

  useEffect(() => {
    if (!playing) return;
    if (t >= n) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => seek(t + 1), 1000 / speed);
    return () => window.clearTimeout(id);
  }, [playing, t, n, speed, seek]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (e.key === "ArrowRight") seek(t + 1);
      else if (e.key === "ArrowLeft") seek(t - 1);
      else if (e.key === "Home") seek(0);
      else if (e.key === "End") seek(n);
      else if (e.key === " ") {
        e.preventDefault();
        setPlaying((p) => (t >= n ? (seek(0), true) : !p));
      } else return;
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [t, n, seek]);

  const step = t < n ? ep.trace[t]! : null;
  const frame = step ? step.frame : ep.final.frame;
  const observation = step ? step.observation : ep.final.observation;

  return (
    <>
      <div className="crumbs">
        <a href={href({ page: "run", run, scope: "all" })}>{run}</a> ·{" "}
        <a href={href({ page: "run", run, scope: task })}>{task}</a> · seed {seed}
      </div>
      <h1 className="episode-title">
        <AgentName agent={ep.agent} />
      </h1>
      <div className="facts">
        <span className="status">
          <span className="dot" style={{ background: ep.success ? "var(--good)" : "var(--critical)" }} aria-hidden="true" />
          {ep.success ? "✓" : "✕"} {outcome(ep)}
        </span>
        <span>
          <strong>{ep.steps}</strong> steps
        </span>
        <span>
          score <strong>{ep.norm !== null ? num2(ep.norm) : "–"}</strong>
        </span>
        <span>
          return <strong>{ep.return.toFixed(3)}</strong>
        </span>
        {ep.invalid > 0 && (
          <span>
            <strong>{ep.invalid}</strong> invalid
          </span>
        )}
        {ep.input_tokens + ep.output_tokens > 0 && (
          <span>
            <strong>{compact(ep.input_tokens + ep.output_tokens)}</strong> tokens
          </span>
        )}
        {ep.cost_usd !== null && (
          <span>
            <strong>{usd(ep.cost_usd)}</strong>
          </span>
        )}
      </div>

      <div className="player">
        <section className="card" style={{ marginBottom: 0 }}>
          {frame ? <GridBoard frame={frame} /> : <pre className="obs">{observation}</pre>}
          <BoardLegend />
          <div className="controls">
            <button className="icon-btn" onClick={() => seek(t - 1)} disabled={t === 0} aria-label="Previous step">
              ←
            </button>
            <button
              className="icon-btn"
              onClick={() => (t >= n ? (seek(0), setPlaying(true)) : setPlaying((p) => !p))}
              aria-label={playing ? "Pause" : "Play"}
              style={{ minWidth: 64 }}
            >
              {playing ? "Pause" : t >= n ? "Replay" : "Play"}
            </button>
            <button className="icon-btn" onClick={() => seek(t + 1)} disabled={t >= n} aria-label="Next step">
              →
            </button>
            <input
              type="range"
              min={0}
              max={n}
              value={t}
              onChange={(e) => seek(Number(e.target.value))}
              aria-label="Step"
            />
            <span className="step-counter">
              {t < n ? `step ${t} / ${n}` : `end (${n} steps)`}
            </span>
            <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Playback speed">
              {SPEEDS.map((s) => (
                <option key={s} value={s}>
                  {s} steps/s
                </option>
              ))}
            </select>
          </div>
          <div className="kbd">← → step · space play/pause · Home / End</div>
        </section>

        <section className="card" style={{ marginBottom: 0 }}>
          {step ? (
            <>
              <div className="note">At step {step.t} the agent chose</div>
              <div className="action">
                {step.action}
                {step.invalid && (
                  <span className="badge" style={{ borderColor: "var(--critical)" }}>
                    <span aria-hidden="true" style={{ color: "var(--critical)" }}>
                      ●
                    </span>
                    invalid action
                  </span>
                )}
                {step.reward > 0 && (
                  <span className="badge" style={{ borderColor: "var(--good)" }}>
                    <span aria-hidden="true" style={{ color: "var(--good)" }}>
                      ●
                    </span>
                    reward {step.reward.toFixed(3)}
                  </span>
                )}
              </div>
              {step.thought && (
                <>
                  <div className="note">{step.input_tokens !== null ? "Reasoning" : "Plan"}</div>
                  <p className="thought">{step.thought}</p>
                </>
              )}
              <div className="meta">
                {step.input_tokens !== null && (
                  <div>
                    tokens in / out
                    <strong>
                      {compact(step.input_tokens)} / {compact(step.output_tokens ?? 0)}
                    </strong>
                  </div>
                )}
                {step.cost_usd !== null && (
                  <div>
                    cost<strong>{usd(step.cost_usd)}</strong>
                  </div>
                )}
                <div>
                  {step.model_latency_ms !== null ? "model latency" : "decision time"}
                  <strong>{ms(step.model_latency_ms ?? step.latency_ms)}</strong>
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="note">End of episode</div>
              <div className="action">{outcome(ep)}</div>
            </>
          )}
          <details>
            <summary>Observation the agent saw</summary>
            <pre className="obs">{observation}</pre>
          </details>
        </section>
      </div>

      <section className="card" style={{ marginTop: 20 }}>
        <h2>Timeline</h2>
        <p className="sub">
          One column per step{ep.trace.some((s) => s.output_tokens) ? "; bars are output tokens" : ""}.
          Click or drag to jump.
        </p>
        <Timeline trace={ep.trace} t={t} onSeek={seek} />
        <div className="legend">
          {ep.trace.some((s) => s.output_tokens) && (
            <span>
              <span className="swatch" style={{ background: "var(--accent)", opacity: 0.55 }} />
              output tokens
            </span>
          )}
          <span>
            <span className="swatch" style={{ background: "var(--critical)" }} />
            invalid action
          </span>
          <span>
            <span className="swatch" style={{ background: "var(--ink)" }} />
            current step
          </span>
        </div>
      </section>
    </>
  );
}
