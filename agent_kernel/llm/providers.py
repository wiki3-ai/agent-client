"""LLM providers — thin presets over :func:`litellm.completion`.

The spec calls for **LiteLLM + Instructor + Pydantic**. This module
contains the LiteLLM layer; all provider logic — request shaping,
auth, error normalization, cost calculation, mocking — is delegated
to LiteLLM. There is one real implementation, :class:`LiteLLMProvider`,
which is a thin wrapper around :func:`litellm.completion`. The other
"providers" exported here are configuration presets on top of it:

- :class:`FakeProvider` — scripts the canonical ``mock_response`` kwarg
  documented at https://docs.litellm.ai/docs/completion/mock_requests.
  Each scripted item is either a JSON string (returned as the assistant
  message) or an :class:`Exception` (raised). No bespoke transport.

- :class:`LMStudioProvider` — preset that routes through LiteLLM's
  built-in ``lm_studio/<model>`` provider against a local LM Studio
  server. By default the model id is auto-detected from
  ``{base_url}/models`` so quickstart / testing works without naming
  the loaded model.

All three classes satisfy the :class:`agent_kernel.llm.adapter.Provider`
Protocol: ``generate_text(messages, response_schema, model) ->
(text, LLMUsage)``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_kernel.llm.adapter import LLMUsage

# Default model used when scripting mock_response — any provider-prefixed id
# that ``litellm.get_llm_provider`` resolves will do; we pick a ubiquitous
# one. The actual network call is intercepted by ``mock_response``.
_MOCK_DEFAULT_MODEL = "gpt-3.5-turbo"


def _extract_usage(
    response: Any,
    *,
    provider_name: str,
    model_for_report: str,
    cost_override_micro: int | None = None,
) -> LLMUsage:
    """Build :class:`LLMUsage` from a LiteLLM ``ModelResponse``.

    Uses :func:`litellm.completion_cost` for cost when not overridden, so
    real-provider pricing comes from LiteLLM's model-cost map, not from
    bespoke arithmetic.
    """
    import litellm

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    if cost_override_micro is not None:
        cost_micro = int(cost_override_micro)
    else:
        try:
            cost_usd = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:
            cost_usd = 0.0
        cost_micro = round(cost_usd * 1_000_000)
    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd_micro=cost_micro,
        provider=provider_name,
        model=model_for_report,
        raw={"model": getattr(response, "model", model_for_report)},
    )


@dataclass
class LiteLLMProvider:
    """Thin :func:`litellm.completion` wrapper.

    All keyword arguments other than ``messages`` / ``model`` are
    forwarded verbatim via ``extra``; this means anything LiteLLM
    accepts (``api_base``, ``api_key``, ``temperature``, ``timeout``,
    ``num_retries``, ``fallbacks``, …) is reachable without
    changes here.
    """

    name: str = "litellm"
    model: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    response_format_json: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def generate_text(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
        model: str | None = None,
    ) -> tuple[str, LLMUsage]:
        import litellm

        used_model = model or self.model
        if not used_model:
            raise ValueError(
                "LiteLLMProvider: no model configured. Set .model or pass "
                "model=… to StructuredLLM(...)."
            )
        kwargs: dict[str, Any] = {
            "model": used_model,
            "messages": messages,
            **self.extra,
        }
        if self.api_base:
            kwargs.setdefault("api_base", self.api_base)
        if self.api_key:
            kwargs.setdefault("api_key", self.api_key)
        if self.response_format_json and "response_format" not in kwargs:
            kwargs["response_format"] = {"type": "json_object"}

        response = litellm.completion(**kwargs)
        text = response.choices[0].message.content or ""
        return text, _extract_usage(
            response, provider_name=self.name, model_for_report=used_model
        )


@dataclass
class FakeProvider:
    """Deterministic, LiteLLM-backed fake for hermetic tests.

    Scripts the canonical ``mock_response`` kwarg per call, exactly as
    documented at https://docs.litellm.ai/docs/completion/mock_requests:

    - Each item in ``script`` is either a string (returned verbatim as
      the assistant message) or an :class:`Exception` (raised).
    - ``handler`` is an alternative: a callable receiving the call
      kwargs and returning the same shape.

    The model id is opaque to the caller (mock_response intercepts the
    network call). We pass a known provider-prefixed id to
    ``litellm.completion`` so its provider-resolution logic is happy.
    """

    name: str = "fake"
    script: list[str | Exception] = field(default_factory=list)
    handler: Callable[..., str | Exception | tuple[str, LLMUsage]] | None = None
    cost_usd_micro_per_call: int = 100  # $0.0001 default — small but non-zero
    _index: int = field(default=0, init=False, repr=False)

    def _next_mock(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
        model: str | None,
    ) -> str | Exception:
        if self.handler is not None:
            result = self.handler(
                messages=messages, response_schema=response_schema, model=model
            )
            if isinstance(result, tuple):
                return result[0]
            return result
        if self._index >= len(self.script):
            raise IndexError(f"FakeProvider script exhausted at call #{self._index + 1}")
        item = self.script[self._index]
        self._index += 1
        return item

    def generate_text(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
        model: str | None = None,
    ) -> tuple[str, LLMUsage]:
        import litellm

        mock = self._next_mock(
            messages=messages, response_schema=response_schema, model=model
        )

        # Pass mock straight through to litellm. Per the docs, a string
        # populates the assistant message; an Exception causes litellm
        # to raise (wrapped as a provider exception).
        try:
            response = litellm.completion(
                model=_MOCK_DEFAULT_MODEL,
                messages=messages,
                mock_response=mock,
            )
        except Exception:
            # If the caller scripted an Exception, propagate the original
            # exception type — that's the semantically useful thing for
            # tests. (litellm wraps it as ``litellm.MockException`` inside
            # a provider error.)
            if isinstance(mock, Exception):
                raise mock from None
            raise

        return response.choices[0].message.content or "", _extract_usage(
            response,
            provider_name=self.name,
            model_for_report=(model or "fake-model"),
            cost_override_micro=self.cost_usd_micro_per_call,
        )


@dataclass
class LMStudioProvider:
    """LiteLLM-backed preset for a local LM Studio server.

    Routes through LiteLLM's built-in ``lm_studio/<model>`` provider.
    LM Studio exposes an OpenAI-compatible API at
    ``http://localhost:1234/v1`` by default; the loaded model is
    advertised at ``/v1/models``.

    If ``model`` is left unset, the first model returned by ``/models``
    is used — so quickstart and ad-hoc testing work with whatever
    model is currently loaded, no configuration required.
    """

    name: str = "lmstudio"
    base_url: str = "http://localhost:1234/v1"
    model: str | None = None
    api_key: str = "lm-studio"  # LM Studio ignores the key but the OpenAI client requires one.
    timeout_s: float = 30.0
    cost_usd_micro_per_call: int | None = 0  # local models are free; override for hosted clones.

    def __post_init__(self) -> None:
        # LiteLLM's lm_studio handler reads LM_STUDIO_API_BASE / LM_STUDIO_API_KEY.
        # Setdefault so explicit env vars from the caller win.
        os.environ.setdefault("LM_STUDIO_API_BASE", self.base_url)
        os.environ.setdefault("LM_STUDIO_API_KEY", self.api_key)

    # ---- model discovery -------------------------------------------------

    def list_models(self) -> list[str]:
        """Return loaded model ids via the OpenAI-compatible ``/models``.

        Returns ``[]`` if the server isn't reachable.
        """
        try:
            from openai import OpenAI
        except ImportError:  # pragma: no cover - openai is a [llm] extra
            return []
        try:
            client = OpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s
            )
            return [m.id for m in client.models.list()]
        except Exception:
            return []

    def is_reachable(self) -> bool:
        return bool(self.list_models())

    def resolve_model(self) -> str:
        """Return a LiteLLM-shaped ``lm_studio/<id>`` model string."""
        chosen = self.model
        if not chosen:
            models = self.list_models()
            if not models:
                raise RuntimeError(
                    f"LM Studio: no models loaded at {self.base_url}/models. "
                    "Load a model in LM Studio or pass model=… explicitly."
                )
            chosen = models[0]
        return chosen if chosen.startswith("lm_studio/") else f"lm_studio/{chosen}"

    # ---- Provider Protocol ----------------------------------------------

    def generate_text(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
        model: str | None = None,
    ) -> tuple[str, LLMUsage]:
        import litellm

        used_model = model or self.resolve_model()
        if not used_model.startswith("lm_studio/"):
            used_model = f"lm_studio/{used_model}"

        # Embed the schema in the system message and ask for json_object —
        # this is the portable LM Studio recipe across the various local
        # backends LM Studio can host. LiteLLM forwards it verbatim.
        import json as _json

        system_msg = {
            "role": "system",
            "content": (
                "Respond with strict JSON that matches this JSON Schema: "
                + _json.dumps(response_schema)
            ),
        }
        response = litellm.completion(
            model=used_model,
            messages=[system_msg, *messages],
            response_format={"type": "json_object"},
            api_base=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_s,
        )
        return response.choices[0].message.content or "", _extract_usage(
            response,
            provider_name=self.name,
            model_for_report=used_model,
            cost_override_micro=self.cost_usd_micro_per_call,
        )
