"""Env-driven configuration for the DSPy LM.

``LiteLLMClient`` is *only* a configuration object now: it reads our
``NOTEBOOK_AGENT_*`` env vars and exposes them in a typed form that
:mod:`notebook_agent.dspy_lm` translates into a :class:`dspy.LM`. There is no
``complete()`` method — every LLM call in the agent goes through DSPy modules.

The fake provider exists so tests can construct a deterministic
:class:`dspy.utils.dummies.DummyLM` without touching LiteLLM at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ENV = "NOTEBOOK_AGENT_MODEL"
DEFAULT_BASE_URL_ENV = "NOTEBOOK_AGENT_BASE_URL"
DEFAULT_API_KEY_ENV = "NOTEBOOK_AGENT_API_KEY"
DEFAULT_PROVIDER_ENV = "NOTEBOOK_AGENT_PROVIDER"
DEFAULT_MAX_TOKENS_ENV = "NOTEBOOK_AGENT_MAX_TOKENS"
DEFAULT_TEMPERATURE_ENV = "NOTEBOOK_AGENT_TEMPERATURE"

DEFAULT_MODEL = "lm_studio/model-name"
DEFAULT_BASE_URL = "http://host.docker.internal:1234/v1"
DEFAULT_API_KEY = "lm-studio"
# Generous default for modern reasoning/thinking models. Chain-of-thought
# can easily consume 4-8k tokens before the final answer is emitted.
DEFAULT_MAX_TOKENS = 16384
DEFAULT_TEMPERATURE = 0.0


@dataclass
class LiteLLMClient:
    """Configuration bag for the DSPy LM.

    Fields are populated from explicit arguments first, then from environment
    variables, then from sensible LM-Studio defaults. Nothing in this class
    issues LM calls — that is DSPy's job.
    """

    provider: str = "lm_studio"
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    lm_calls_log: Path | None = None
    # For the "fake" provider, ``fake_answers`` is a list of dicts mapping
    # signature output-field names to canned values, as ``DummyLM`` consumes.
    fake_answers: list[dict[str, Any]] | None = None
    # Convenience: when callers only have a single string they want every
    # call to "produce", they can set ``fake_response`` and the bridge will
    # wrap it as ``[{"answer": fake_response}]``.
    fake_response: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        lm_calls_log: Path | str | None = None,
        fake_answers: list[dict[str, Any]] | None = None,
        fake_response: str | None = None,
        **extra: Any,
    ) -> None:
        prov_env = os.environ.get(DEFAULT_PROVIDER_ENV)
        self.provider = (provider or prov_env or "lm_studio").lower()
        self.model = model or os.environ.get(DEFAULT_MODEL_ENV) or DEFAULT_MODEL
        self.base_url = base_url or os.environ.get(DEFAULT_BASE_URL_ENV) or DEFAULT_BASE_URL
        self.api_key = api_key or os.environ.get(DEFAULT_API_KEY_ENV) or DEFAULT_API_KEY
        self.max_tokens = _coerce_int(
            max_tokens, os.environ.get(DEFAULT_MAX_TOKENS_ENV), DEFAULT_MAX_TOKENS
        )
        self.temperature = _coerce_float(
            temperature, os.environ.get(DEFAULT_TEMPERATURE_ENV), DEFAULT_TEMPERATURE
        )
        self.lm_calls_log = Path(lm_calls_log) if lm_calls_log else None
        self.fake_answers = fake_answers
        self.fake_response = fake_response
        self.extra = dict(extra)

    def is_fake(self) -> bool:
        return self.provider == "fake"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # api_key is intentionally omitted.
            "lm_calls_log": str(self.lm_calls_log) if self.lm_calls_log else None,
        }


def _coerce_int(explicit: int | None, env_val: str | None, default: int) -> int:
    if explicit is not None:
        return int(explicit)
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            pass
    return default


def _coerce_float(explicit: float | None, env_val: str | None, default: float) -> float:
    if explicit is not None:
        return float(explicit)
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass
    return default


def write_lm_call_log(path: Path | str | None, payload: dict[str, Any]) -> None:
    """Append a single JSON line to *path* (used by the agent run record).

    ``api_key`` is stripped if present; this writer never accepts a payload
    that would leak credentials.
    """
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    safe = {k: v for k, v in payload.items() if k != "api_key"}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe, default=str) + "\n")


__all__ = [
    "DEFAULT_API_KEY",
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "LiteLLMClient",
    "write_lm_call_log",
]
