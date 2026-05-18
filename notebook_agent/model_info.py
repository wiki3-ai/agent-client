"""Provider-side model metadata lookup.

LM Studio (and most OpenAI-compatible servers) expose ``GET /v1/models``
and a per-model record that includes details like the model's context
window. This module wraps that endpoint so skill notebooks can ground
their parameter choices (``max_tokens``, etc.) in what the provider
actually advertises, rather than hard-coding values.

LM Studio specifically exposes a richer ``/api/v0/models/{id}`` endpoint
with fields like ``loaded_context_length`` and ``max_context_length``.
We probe that first and fall back to the standard OpenAI shape.

Cached per ``(base_url, model)`` for the lifetime of the process so that
skill notebooks calling this on every run don't hammer the provider.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .litellm_client import LiteLLMClient


@dataclass
class ModelInfo:
    """Provider-reported metadata for one model."""

    id: str
    base_url: str
    raw: dict[str, Any]
    context_length: int | None = None
    loaded_context_length: int | None = None
    state: str | None = None
    family: str | None = None
    arch: str | None = None

    def recommended_max_tokens(self, *, headroom: int = 1024) -> int | None:
        """A reasonable default for ``max_tokens`` given the model's window.

        Returns ``loaded_context_length - headroom`` (clamped to a minimum
        of 4096) when known, else ``None``. Callers should treat ``None``
        as "no provider hint — use the user's configured default".
        """
        n = self.loaded_context_length or self.context_length
        if not n:
            return None
        return max(4096, int(n) - int(headroom))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "base_url": self.base_url,
            "context_length": self.context_length,
            "loaded_context_length": self.loaded_context_length,
            "state": self.state,
            "family": self.family,
            "arch": self.arch,
            "recommended_max_tokens": self.recommended_max_tokens(),
        }


_CACHE: dict[tuple[str, str], ModelInfo] = {}


def _http_get_json(url: str, *, api_key: str | None = None, timeout: float = 5.0) -> Any:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - controlled URL
        return json.loads(resp.read().decode("utf-8"))


def _strip_provider_prefix(model: str) -> str:
    """``lm_studio/google/gemma-4-31b`` → ``google/gemma-4-31b``."""
    return model.split("/", 1)[1] if "/" in model else model


def model_info(
    client_or_base_url: LiteLLMClient | str | None = None,
    model: str | None = None,
    *,
    refresh: bool = False,
) -> ModelInfo | None:
    """Look up provider-reported metadata for ``model``.

    Accepts either a :class:`LiteLLMClient` (uses its ``base_url`` /
    ``api_key`` / ``model``) or a raw ``base_url`` + ``model``. Returns
    ``None`` if the provider is unreachable or doesn't recognise the model.
    """
    if isinstance(client_or_base_url, LiteLLMClient):
        base_url = client_or_base_url.base_url
        api_key = client_or_base_url.api_key
        model = model or client_or_base_url.model
    elif client_or_base_url is None:
        c = LiteLLMClient()
        base_url, api_key = c.base_url, c.api_key
        model = model or c.model
    else:
        base_url = client_or_base_url
        api_key = None
    if not model:
        return None

    model_id = _strip_provider_prefix(model)
    key = (base_url.rstrip("/"), model_id)
    if not refresh and key in _CACHE:
        return _CACHE[key]

    base = base_url.rstrip("/")
    # LM Studio's richer per-model endpoint sits under /api/v0/. Try it
    # first; fall back to the OpenAI-shaped list at /v1/models.
    raw: dict[str, Any] | None = None
    candidates = [
        f"{base.rsplit('/v1', 1)[0]}/api/v0/models/{model_id}",
        f"{base}/models/{model_id}",
    ]
    for url in candidates:
        try:
            data = _http_get_json(url, api_key=api_key)
        except (URLError, TimeoutError, ValueError, OSError):
            continue
        if isinstance(data, dict) and ("id" in data or "object" in data):
            raw = data
            break
    if raw is None:
        # Last resort: list /v1/models and pick a matching id.
        try:
            listing = _http_get_json(f"{base}/models", api_key=api_key)
        except (URLError, TimeoutError, ValueError, OSError):
            return None
        items = listing.get("data") if isinstance(listing, dict) else None
        if not isinstance(items, list):
            return None
        match = next(
            (it for it in items if isinstance(it, dict) and it.get("id") == model_id),
            None,
        )
        if match is None:
            return None
        raw = match

    info = ModelInfo(
        id=str(raw.get("id") or model_id),
        base_url=base,
        raw=raw,
        context_length=_first_int(raw, "max_context_length", "context_length", "context_window"),
        loaded_context_length=_first_int(raw, "loaded_context_length"),
        state=raw.get("state") if isinstance(raw.get("state"), str) else None,
        family=raw.get("family") if isinstance(raw.get("family"), str) else None,
        arch=raw.get("arch") if isinstance(raw.get("arch"), str) else None,
    )
    _CACHE[key] = info
    return info


def _first_int(d: dict[str, Any], *keys: str) -> int | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    return None


def clear_model_info_cache() -> None:
    """Drop the in-process cache (mainly for tests)."""
    _CACHE.clear()


__all__ = [
    "ModelInfo",
    "clear_model_info_cache",
    "model_info",
]
