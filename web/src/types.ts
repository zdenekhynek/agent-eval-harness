// Mirrors the documents built by src/harness/views.py.

export type Interval = [number, number];

export interface RunSummary {
  id: string;
  created_at: string;
  agents: string[];
  tasks: string[];
  episodes: number;
  cost_usd: number | null;
  leaderboard: string;
}

export interface VsNext {
  agent: string;
  pairs: number;
  diff: number;
  diff_ci: Interval;
  /** ahead: CI excludes 0 and |diff| >= min_effect; marginal: real but smaller. */
  status: "ahead" | "marginal" | "tied";
  min_effect: number;
}

export interface Row {
  rank: number;
  agent: string;
  n: number;
  norm: number | null;
  norm_ci: Interval | null;
  success: number;
  success_ci: Interval | null;
  mean_return: number;
  steps_on_success: number | null;
  invalid_rate: number;
  tokens: number | null;
  cost_usd: number | null;
  vs_next: VsNext | null;
}

export interface Scope {
  id: string; // "all" or a task id
  label: string;
  rows: Row[];
  excluded: string[];
}

export interface EpisodeRow {
  agent: string;
  task: string;
  seed: number;
  success: boolean;
  truncated: boolean;
  steps: number;
  return: number;
  norm: number | null;
  invalid: number;
  tokens: number;
  cost_usd: number | null;
  path: string;
}

export interface Leaderboard {
  run_id: string;
  tasks: string[];
  agents: string[];
  references: Record<string, { floor: number; ceiling: number; n: number }>;
  scopes: Scope[];
  episodes: EpisodeRow[];
}

export interface Frame {
  grid: string[];
  agent: [number, number] | null;
  direction: number | null;
  carrying: boolean;
  mission: string;
}

export interface Step {
  t: number;
  action: string;
  thought: string | null;
  reward: number;
  invalid: boolean;
  input_tokens: number | null;
  output_tokens: number | null;
  cost_usd: number | null;
  latency_ms: number;
  model_latency_ms: number | null;
  observation: string;
  frame: Frame | null;
}

export interface EpisodeDoc {
  run_id: string;
  agent: string;
  task: string;
  seed: number;
  success: boolean;
  truncated: boolean;
  steps: number;
  return: number;
  norm: number | null;
  invalid: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number | null;
  duration_ms: number;
  trace: Step[];
  final: { observation: string; frame: Frame | null };
}
