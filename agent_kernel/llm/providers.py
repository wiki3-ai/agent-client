"""Concrete LLM providers.

- ``FakeProvider``: deterministic in-process provider used by CI integration
  tests. Returns JSON strings provided by a per-call ``script`` or by a
  callback function. Useful for exercising retry-on-validation by scripting
  an invalid response followed by a valid one.

- ``LMStudioProvider``: OpenAI-compatible HTTP provider that targets a
  local LM Studio server. Provides ``is_reachable()`` so tests can skip
  cleanly when the server isn't running.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_kernel.llm.adapter import LLMUsage


@dataclass
class FakeProvider:
    """Deterministic provider for hermetic tests.

    Specify behavior with one of:
    - ``script``: a list of raw JSON strings (or exception instances) to
      return in order, one per call.
    - ``handler``: a callable taking the call kwargs and returning either a
      string or ``(string, LLMUsage)``.

    Each call also returns a ``LLMUsage`` derived from ``cost_usd_micro_per_call``
    and the rendered text length (proxy for completion tokens).
    """

    name: str = "fake"
    script: list[str | Exception] = field(default_factory=list)
    handler: Callable[..., str | tuple[str, LLMUsage]] | None = None
    cost_usd_micro_per_call: int = 100  # $0.0001 default — small but non-zero
    _index: int = field(default=0, init=False, repr=False)

    def generate_text(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
        model: str | None = None,
    ) -> tuple[str, LLMUsage]:
        # If a handler is provided, prefer it.
        if self.handler is not None:
            result = self.handler(messages=messages, response_schema=response_schema, model=model)
            if isinstance(result, tuple):
                return result
            text = result
        else:
            if self._index >= len(self.script):
                raise IndexError(f"FakeProvider script exhausted at call #{self._index + 1}")
            item = self.script[self._index]
            self._index += 1
            if isinstance(item, Exception):
                raise item
            text = item

        usage = LLMUsage(
            prompt_tokens=sum(len(m.get("content", "").split()) for m in messages),
            completion_tokens=max(1, len(text.split())),
            cost_usd_micro=self.cost_usd_micro_per_call,
            provider=self.name,
            model=model or "fake-model",
            raw={"messages": len(messages)},
        )
        return text, usage


@dataclass
class LMStudioProvider:
    """Provider that talks to a local LM Studio server (OpenAI-compatible).

    LM Studio exposes ``http://localhost:1234/v1/chat/completions`` by
    default. Tests using this provider should call ``is_reachable()`` first
    and skip if False; the ``llm`` pytest marker should also be applied.
    """

    name: str = "lmstudio"
    base_url: str = "http://localhost:1234/v1"
    default_model: str = "local-model"
    timeout_s: float = 30.0
    cost_usd_micro_per_call: int = 0  # local models are free

    def is_reachable(self) -> bool:
        try:
            import urllib.error
            import urllib.request

            req = urllib.request.Request(f"{self.base_url}/models")
            with urllib.request.urlopen(req, timeout=2) as r:
                return r.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def generate_text(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
        model: str | None = None,
    ) -> tuple[str, LLMUsage]:
        import urllib.request

        model_name = model or self.default_model
        # Force JSON output via the OpenAI ``response_format`` extension.
        body = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Respond with strict JSON matching this schema: "
                        + json.dumps(response_schema)
                    ),
                },
                *messages,
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            payload = json.loads(r.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"]
        u = payload.get("usage", {}) or {}
        return text, LLMUsage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            cost_usd_micro=self.cost_usd_micro_per_call,
            provider=self.name,
            model=model_name,
            raw=payload,
        )
