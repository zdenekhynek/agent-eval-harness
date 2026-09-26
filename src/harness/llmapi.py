"""Minimal Anthropic Messages API client with token and cost accounting.

Written against the HTTP API with the standard library only, so the harness
keeps zero runtime dependencies. The transport is injectable, which is how the
test suite replays canned responses without a network or an API key.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

API_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})

# USD per million tokens (input, output), matched by model-name prefix.
# Check https://www.anthropic.com/pricing before trusting cost figures.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25

# (url, headers, body) -> (status, response headers, response body)
Transport = Callable[[str, dict[str, str], bytes], tuple[int, dict[str, str], bytes]]


class MissingAPIKeyError(RuntimeError):
    pass


class APIError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"Anthropic API error {status}: {message}")
        self.status = status


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Usage:
        return cls(
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
            cache_read_input_tokens=int(data.get("cache_read_input_tokens") or 0),
            cache_creation_input_tokens=int(data.get("cache_creation_input_tokens") or 0),
        )


def price_for(model: str) -> tuple[float, float] | None:
    for prefix, price in sorted(PRICES.items(), key=lambda kv: -len(kv[0])):
        if model.startswith(prefix):
            return price
    return None


def cost_usd(model: str, usage: Usage) -> float | None:
    """Dollar cost of one call, or None for a model without a known price."""
    price = price_for(model)
    if price is None:
        return None
    inp, out = price
    input_equiv = (
        usage.input_tokens
        + usage.cache_read_input_tokens * CACHE_READ_MULTIPLIER
        + usage.cache_creation_input_tokens * CACHE_WRITE_MULTIPLIER
    )
    return (input_equiv * inp + usage.output_tokens * out) / 1_000_000


@dataclass
class Response:
    content: list[dict[str, Any]]
    usage: Usage
    stop_reason: str | None
    model: str
    latency_ms: float
    retries: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def text(self) -> str:
        return "\n".join(b["text"] for b in self.content if b.get("type") == "text").strip()

    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]


def urllib_transport(
    url: str, headers: dict[str, str], body: bytes
) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers or {}), err.read()


class AnthropicClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        transport: Transport | None = None,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingAPIKeyError(
                "set ANTHROPIC_API_KEY to run LLM agents (baselines need no key)"
            )
        self.api_key = key
        self.base_url = (
            base_url or os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.transport = transport or urllib_transport
        self.max_retries = max_retries
        self.sleep = sleep

    def messages(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> Response:
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        body = json.dumps(payload).encode()
        url = f"{self.base_url}/v1/messages"

        started = time.perf_counter()
        attempt = 0
        while True:
            try:
                status, resp_headers, resp_body = self.transport(url, headers, body)
            except (urllib.error.URLError, TimeoutError) as err:
                if attempt >= self.max_retries:
                    raise APIError(0, f"network error: {err}") from err
                self.sleep(self._backoff(attempt, None))
                attempt += 1
                continue

            if status == 200:
                data = json.loads(resp_body)
                return Response(
                    content=list(data.get("content") or []),
                    usage=Usage.from_api(data.get("usage") or {}),
                    stop_reason=data.get("stop_reason"),
                    model=str(data.get("model") or model),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    retries=attempt,
                    raw=data,
                )
            if status in RETRY_STATUSES and attempt < self.max_retries:
                self.sleep(self._backoff(attempt, _header(resp_headers, "retry-after")))
                attempt += 1
                continue
            raise APIError(status, _error_message(resp_body))

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        return min(2.0**attempt, 30.0)


def _header(headers: dict[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None


def _error_message(body: bytes) -> str:
    try:
        data = json.loads(body)
        return str(data.get("error", {}).get("message") or data)
    except (ValueError, AttributeError):
        return body.decode(errors="replace")[:500]
