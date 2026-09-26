"""Per-step and per-episode records. Everything the dashboard shows comes from here."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StepRecord:
    t: int
    observation: str
    action: str
    reward: float
    thought: str | None = None
    invalid: bool = False
    latency_ms: float = 0.0
    # Filled in by LLM agents from milestone 1; None for baselines.
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@dataclass
class Episode:
    run_id: str
    agent: str
    task: str
    seed: int
    success: bool = False
    steps: int = 0
    total_return: float = 0.0
    truncated: bool = False
    invalid_actions: int = 0
    duration_ms: float = 0.0
    final_observation: str = ""
    trace: list[StepRecord] = field(default_factory=list)

    def trace_dicts(self) -> list[dict[str, Any]]:
        return [asdict(step) for step in self.trace]
