"""LiteLLM client wrapper (Section 14.11).

Wraps :mod:`litellm` so the rest of the agent has a single entry point for
LLM calls. The wrapper:

* reads configuration from environment variables (with sane defaults pointing
  at LM Studio);
* exposes a ``FakeProvider`` that uses LiteLLM's built-in ``mock_response``
  feature so unit and integration tests can run without any LLM service;
* logs every call to ``logs/lm_calls.jsonl`` when given a log path.

The agent must run without LM Studio for non-generation milestones, so this
module never imports LiteLLM at module load time.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ENV = "NOTEBOOK_AGENT_MODEL"
DEFAULT_BASE_URL_ENV = "NOTEBOOK_AGENT_BASE_URL"
DEFAULT_API_KEY_ENV = "NOTEBOOK_AGENT_API_KEY"
DEFAULT_PROVIDER_ENV = "NOTEBOOK_AGENT_PROVIDER"

DEFAULT_MODEL = "lm_studio/model-name"
DEFAULT_BASE_URL = "http://host.docker.internal:1234/v1"
DEFAULT_API_KEY = "lm-studio"


class LLMUnavailableError(RuntimeError):
    """Raised when an LLM call cannot be made (missing config or network)."""


@dataclass
class LLMResponse:
    text: str
    raw: dict[str, Any]
    provider: str
    model: str
    latency_seconds: float


def _now_iso() -> str:
    from ._clock import iso_now

    return iso_now()


def _redact(s: str | None) -> str | None:
    if not s:
        return s
    return "<redacted>"


def _append_lm_call_log(path: Path | str | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    safe = dict(payload)
    safe.pop("api_key", None)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe, default=str) + "\n")


class LiteLLMClient:
    """Thin wrapper around :func:`litellm.completion`.

    ``provider`` selects between ``"lm_studio"`` (default), ``"fake"`` (for
    tests, uses ``mock_response``), or ``"auto"`` which reads
    ``NOTEBOOK_AGENT_PROVIDER`` from the environment and falls back to
    ``"lm_studio"``.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        lm_calls_log: Path | str | None = None,
        fake_response: str | None = None,
    ) -> None:
        prov_env = os.environ.get(DEFAULT_PROVIDER_ENV)
        self.provider = (provider or prov_env or "lm_studio").lower()
        self.model = model or os.environ.get(DEFAULT_MODEL_ENV) or DEFAULT_MODEL
        self.base_url = base_url or os.environ.get(DEFAULT_BASE_URL_ENV) or DEFAULT_BASE_URL
        self.api_key = api_key or os.environ.get(DEFAULT_API_KEY_ENV) or DEFAULT_API_KEY
        self.lm_calls_log = Path(lm_calls_log) if lm_calls_log else None
        self.fake_response = fake_response

    # ------------------------------------------------------------------

    def is_fake(self) -> bool:
        return self.provider == "fake"

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        fake_response: str | None = None,
    ) -> LLMResponse:
        """Run a single completion. Returns a normalized :class:`LLMResponse`.

        For the ``fake`` provider, ``fake_response`` (or the client default)
        is returned via LiteLLM's ``mock_response`` kwarg, which makes the
        call deterministic and does not touch any network.
        """
        try:
            import litellm  # type: ignore[import-untyped]
        except ImportError as e:  # pragma: no cover - litellm always installed in this env
            raise LLMUnavailableError("litellm is not installed") from e

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if self.provider == "fake":
            mock = fake_response if fake_response is not None else (self.fake_response or "")
            kwargs["mock_response"] = mock
        elif self.provider == "lm_studio":
            kwargs["api_base"] = self.base_url
            kwargs["api_key"] = self.api_key
        else:
            # Treat any other provider name as opaque (e.g. "openai/gpt-4").
            kwargs["api_base"] = self.base_url
            kwargs["api_key"] = self.api_key

        start = time.monotonic()
        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize all provider errors
            _append_lm_call_log(
                self.lm_calls_log,
                {
                    "ts": _now_iso(),
                    "provider": self.provider,
                    "model": self.model,
                    "api_base": _redact(self.base_url) if self.provider != "fake" else None,
                    "prompt_chars": len(prompt),
                    "system_chars": len(system or ""),
                    "success": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
            raise LLMUnavailableError(f"LLM call failed: {exc!s}") from exc
        latency = time.monotonic() - start

        # Normalize text out of the OpenAI-shaped response.
        text = ""
        raw_dict: dict[str, Any] = {}
        try:
            raw_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        except Exception:
            raw_dict = {}
        try:
            msg = response.choices[0].message
            text = msg.content or ""
            if not text:
                # Fallback for thinking models that put output in reasoning_content
                text = getattr(msg, "reasoning_content", None) or ""
        except Exception:
            text = ""

        _append_lm_call_log(
            self.lm_calls_log,
            {
                "ts": _now_iso(),
                "provider": self.provider,
                "model": self.model,
                "api_base": _redact(self.base_url) if self.provider != "fake" else None,
                "prompt_chars": len(prompt),
                "system_chars": len(system or ""),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "success": True,
                "latency_seconds": latency,
                "response_chars": len(text),
            },
        )
        return LLMResponse(
            text=text,
            raw=raw_dict,
            provider=self.provider,
            model=self.model,
            latency_seconds=latency,
        )
