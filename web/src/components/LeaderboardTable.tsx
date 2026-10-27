import { ci2, ciPct, compact, num2, pct, signed2, signedCi, usd } from "../format";
import { linear, normDomain } from "../scale";
import type { Scope } from "../types";
import { AgentName } from "./AgentName";
import { CI_WIDTH, CIAxis, CIBar } from "./CIBar";

const STATUS_LABEL = { ahead: "ahead", marginal: "marginally ahead", tied: "tied" } as const;

interface Props {
  scope: Scope;
  hot: string | null;
  onHover: (agent: string | null) => void;
}

export function LeaderboardTable({ scope, hot, onHover }: Props) {
  const rows = scope.rows;
  const x = linear(
    normDomain(rows.flatMap((r) => (r.norm_ci ? [r.norm_ci[0], r.norm_ci[1]] : [r.norm]))),
    [8, CI_WIDTH - 8],
  );
  const pooled = scope.id === "all";

  return (
    <div className="table-wrap">
      <table className="board">
        <thead>
          <tr>
            <th className="num">#</th>
            <th>Agent</th>
            <th>Normalised score, 95% CI</th>
            <th className="num">Score</th>
            <th className="num">Success</th>
            <th className="num">Steps</th>
            <th className="num">Invalid</th>
            <th className="num">Tokens/ep</th>
            <th className="num">$/ep</th>
            <th>Versus the next rank</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={r.agent}
              className={hot === r.agent ? "hot" : undefined}
              onPointerEnter={() => onHover(r.agent)}
              onPointerLeave={() => onHover(null)}
            >
              <td className="num">{r.rank}</td>
              <td>
                <AgentName agent={r.agent} />
              </td>
              <td>
                {r.norm !== null && r.norm_ci ? (
                  <CIBar x={x} value={r.norm} ci={r.norm_ci} />
                ) : (
                  <span className="note">no reference</span>
                )}
              </td>
              <td className="num" title={r.norm_ci ? `95% CI ${ci2(r.norm_ci)}` : undefined}>
                {r.norm !== null ? num2(r.norm) : "–"}
                {r.norm_ci && <div className="note">{ci2(r.norm_ci)}</div>}
              </td>
              <td className="num" title={pooled ? "mean of per-task success rates" : undefined}>
                {pct(r.success)}
                {r.success_ci && <div className="note">{ciPct(r.success_ci)}</div>}
              </td>
              <td className="num">{r.steps_on_success !== null ? r.steps_on_success.toFixed(1) : "–"}</td>
              <td className="num">{r.invalid_rate ? `${(r.invalid_rate * 100).toFixed(1)}%` : "0%"}</td>
              <td className="num">{r.tokens !== null ? compact(r.tokens) : "–"}</td>
              <td className="num">{usd(r.cost_usd)}</td>
              <td className="vs">
                {r.vs_next && (
                  <>
                    <strong>{STATUS_LABEL[r.vs_next.status]}</strong>
                    {r.vs_next.status === "tied" ? " · " : " by "}
                    {signed2(r.vs_next.diff)}{" "}
                    <span className="note">{signedCi(r.vs_next.diff_ci)}</span>
                  </>
                )}
              </td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <td />
            <td />
            <td>
              <CIAxis x={x} />
            </td>
            <td colSpan={7} />
          </tr>
        </tfoot>
      </table>
    </div>
  );
}
