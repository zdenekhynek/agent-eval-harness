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
    latency_ms: float = 0.0  # whole agent decision, wall clock
    # Filled in by LLM agents; None for baselines.
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model_latency_ms: float | None = None  # API round trip(s), including retries


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
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # None when no step reported a cost
    trace: list[StepRecord] = field(default_factory=list)

    def add_usage(self, step: StepRecord) -> None:
        self.input_tokens += step.input_tokens or 0
        self.output_tokens += step.output_tokens or 0
        if step.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + step.cost_usd

    def trace_dicts(self) -> list[dict[str, Any]]:
        return [asdict(step) for step in self.trace]
