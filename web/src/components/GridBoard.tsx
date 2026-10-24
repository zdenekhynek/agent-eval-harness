import type { Frame } from "../types";

// The gridworld as the agent saw it at one step, drawn from the text map in
// its observation. Each object has a shape as well as a colour (flag, waves,
// key, padlock), and the agent is ink, so nothing relies on hue alone.
const C = 40; // cell size in viewBox units
const AXIS = 16;

export function GridBoard({ frame }: { frame: Frame }) {
  const rows = frame.grid;
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const size = { w: w * C + AXIS, h: h * C + AXIS };
  return (
    <svg
      className="board-svg"
      viewBox={`0 0 ${size.w} ${size.h}`}
      role="img"
      aria-label={describe(frame)}
    >
      <g transform={`translate(${AXIS} ${AXIS})`}>
        {rows.map((row, y) =>
          [...row].map((glyph, x) => <Cell key={`${x},${y}`} glyph={glyph} x={x * C} y={y * C} />),
        )}
        {frame.agent && frame.direction !== null && (
          <Agent x={frame.agent[0]} y={frame.agent[1]} direction={frame.direction} carrying={frame.carrying} />
        )}
      </g>
      {Array.from({ length: w }, (_, x) => (
        <text key={`x${x}`} x={AXIS + x * C + C / 2} y={11} textAnchor="middle">
          {x}
        </text>
      ))}
      {Array.from({ length: h }, (_, y) => (
        <text key={`y${y}`} x={AXIS / 2} y={AXIS + y * C + C / 2 + 4} textAnchor="middle">
          {y}
        </text>
      ))}
    </svg>
  );
}

function Cell({ glyph, x, y }: { glyph: string; x: number; y: number }) {
  const floor = <rect x={x} y={y} width={C} height={C} fill="var(--floor)" stroke="var(--floor-line)" />;
  const c = { x: x + C / 2, y: y + C / 2 };
  switch (glyph) {
    case "#":
      return <rect x={x} y={y} width={C} height={C} fill="var(--wall)" stroke="var(--surface)" strokeWidth={2} />;
    case "G":
      return (
        <g>
          {floor}
          <rect x={x + 5} y={y + 5} width={C - 10} height={C - 10} rx={4} fill="var(--goal)" />
          <path d={`M${c.x - 6} ${c.y + 9} V${c.y - 9} L${c.x + 8} ${c.y - 4} L${c.x - 6} ${c.y + 1}`} fill="#fff" stroke="#fff" strokeWidth={1.5} strokeLinejoin="round" />
        </g>
      );
    case "~":
      return (
        <g>
          <rect x={x} y={y} width={C} height={C} fill="var(--lava)" stroke="var(--surface)" strokeWidth={2} />
          {[-7, 3].map((dy) => (
            <path
              key={dy}
              d={`M${x + 7} ${c.y + dy} q4 -5 8 0 t8 0 t8 0`}
              fill="none"
              stroke="#fff"
              strokeWidth={2}
              strokeLinecap="round"
              opacity={0.85}
            />
          ))}
        </g>
      );
    case "K":
      return (
        <g>
          {floor}
          <circle cx={c.x - 6} cy={c.y} r={6} fill="none" stroke="var(--key)" strokeWidth={3} />
          <path d={`M${c.x} ${c.y} H${c.x + 12} M${c.x + 8} ${c.y} v5 M${c.x + 12} ${c.y} v5`} stroke="var(--key)" strokeWidth={3} strokeLinecap="round" />
        </g>
      );
    case "L":
    case "D":
      return (
        <g>
          <rect x={x + 2} y={y + 2} width={C - 4} height={C - 4} rx={3} fill="var(--surface-2)" stroke="var(--door)" strokeWidth={2} />
          {glyph === "L" ? (
            <g stroke="var(--door)" strokeWidth={2} fill="none">
              <rect x={c.x - 6} y={c.y - 1} width={12} height={10} rx={2} fill="var(--door)" />
              <path d={`M${c.x - 4} ${c.y - 1} v-4 a4 4 0 0 1 8 0 v4`} />
            </g>
          ) : (
            <circle cx={c.x + 8} cy={c.y} r={2.5} fill="var(--door)" />
          )}
        </g>
      );
    case "_":
      return (
        <g>
          {floor}
          <rect x={x + 2} y={y + 2} width={5} height={C - 4} rx={2} fill="var(--door)" />
          <rect x={x + C - 7} y={y + 2} width={5} height={C - 4} rx={2} fill="var(--door)" />
        </g>
      );
    default:
      return floor;
  }
}

function Agent({ x, y, direction, carrying }: { x: number; y: number; direction: number; carrying: boolean }) {
  return (
    <g className="agent" style={{ transform: `translate(${x * C + C / 2}px, ${y * C + C / 2}px)` }}>
      <g transform={`rotate(${direction * 90})`}>
        <path d="M13 0 L-9 -11 L-4 0 L-9 11 Z" fill="var(--ink)" stroke="var(--surface)" strokeWidth={2} strokeLinejoin="round" />
      </g>
      {carrying && <circle cx={11} cy={-11} r={5} fill="var(--key)" stroke="var(--surface)" strokeWidth={2} />}
    </g>
  );
}

const DIRS = ["east", "south", "west", "north"];

function describe(frame: Frame): string {
  if (!frame.agent || frame.direction === null) return "Gridworld map";
  const [x, y] = frame.agent;
  return `Agent at (${x}, ${y}) facing ${DIRS[frame.direction]}${frame.carrying ? ", carrying the key" : ""}`;
}

export function BoardLegend() {
  const item = (label: string, swatch: React.ReactNode) => (
    <span>
      <svg width={14} height={14} viewBox="0 0 14 14" style={{ verticalAlign: -2, marginRight: 6 }} aria-hidden="true">
        {swatch}
      </svg>
      {label}
    </span>
  );
  return (
    <div className="legend">
      {item("you", <path d="M13 7 L2 1 L4.5 7 L2 13 Z" fill="var(--ink)" />)}
      {item("goal", <rect width={14} height={14} rx={3} fill="var(--goal)" />)}
      {item("lava", <rect width={14} height={14} rx={3} fill="var(--lava)" />)}
      {item("key", <circle cx={7} cy={7} r={4.5} fill="none" stroke="var(--key)" strokeWidth={3} />)}
      {item("door", <rect x={1} y={1} width={12} height={12} rx={2} fill="none" stroke="var(--door)" strokeWidth={2} />)}
      {item("wall", <rect width={14} height={14} rx={2} fill="var(--wall)" />)}
    </div>
  );
}
