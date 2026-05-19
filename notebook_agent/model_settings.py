"""Per-model settings cached in a Papermill-parameterized notebook.

The agent learns model-specific settings (most importantly the value
space for ``reasoning_effort``, which varies per server/model) by running
its own Retrieve→Compose→Transform→Generate loop on a discovery skill.
The result is persisted to a Papermill-style notebook on disk:

    sessions/<model-slug>/model_settings.ipynb

The notebook's *parameters cell* (tagged ``parameters``, per Papermill
convention) is the canonical record — re-executing the notebook
re-validates the settings against the live server. Other cells contain
the probe code and human-readable findings.

This module is the small Python surface around that notebook: detecting
which model is currently loaded by the server, computing a slug,
locating the notebook, and reading/writing its parameters cell. It
deliberately does **not** run the discovery itself — that's the job of
the ``core.discover_model_settings`` skill.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .litellm_client import LiteLLMClient

_SLUG_RE = re.compile(r"[^a-z0-9._-]+")


def model_slug(model_id: str) -> str:
    """Map a model id (``lm_studio/google/gemma-4-31b``) to a filesystem slug.

    Collapses provider prefixes and any non-portable characters to ``-``.
    """
    # Drop any provider prefix (``lm_studio/...``); keep the publisher/name.
    parts = model_id.split("/")
    if len(parts) >= 2 and parts[0] in {"lm_studio", "openai", "anthropic", "ollama"}:
        parts = parts[1:]
    flat = "/".join(parts).lower()
    return _SLUG_RE.sub("-", flat).strip("-")


def settings_notebook_path(model_id: str, sessions_root: Path | str = "sessions") -> Path:
    """Return the on-disk path for *model_id*'s settings notebook."""
    return Path(sessions_root) / model_slug(model_id) / "model_settings.ipynb"


# ---------------------------------------------------------------------------
# Loaded-models detection
# ---------------------------------------------------------------------------


@dataclass
class LoadedModel:
    """One model the provider reports as currently loaded."""

    id: str
    raw: dict[str, Any]
    context_length: int | None = None
    state: str | None = None


def _http_get_json(url: str, *, api_key: str | None = None, timeout: float = 5.0) -> Any:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - controlled URL
        return json.loads(resp.read().decode("utf-8"))


def loaded_models(client: LiteLLMClient | None = None) -> list[LoadedModel]:
    """Return the list of models the provider reports as currently loaded.

    For LM Studio this queries ``/api/v0/models`` (the extended endpoint
    that reports a ``state`` field) and filters to ``state == "loaded"``.
    Falls back to the OpenAI-compatible ``/v1/models`` listing, treating
    everything it returns as loaded (which matches LM Studio's default
    behavior — its ``/v1/models`` only lists currently-loaded models).

    Returns an empty list if the server is unreachable.
    """
    c = client or LiteLLMClient()
    base = c.base_url.rstrip("/")
    api_key = c.api_key
    root = base.rsplit("/v1", 1)[0]
    # Try LM Studio's extended listing first.
    out: list[LoadedModel] = []
    try:
        data = _http_get_json(f"{root}/api/v0/models", api_key=api_key)
    except (URLError, TimeoutError, ValueError, OSError):
        data = None
    if isinstance(data, dict):
        items = data.get("data") if isinstance(data.get("data"), list) else None
        for it in items or []:
            if not isinstance(it, dict):
                continue
            state = it.get("state") if isinstance(it.get("state"), str) else None
            # On /api/v0/models, only include those reported as loaded.
            if state and state != "loaded":
                continue
            out.append(
                LoadedModel(
                    id=str(it.get("id") or ""),
                    raw=it,
                    context_length=_first_int(it, "loaded_context_length", "max_context_length"),
                    state=state,
                )
            )
    if out:
        return [m for m in out if m.id]
    # Fall back to OpenAI-shaped /v1/models. LM Studio's /v1/models only
    # returns loaded models in practice.
    try:
        data = _http_get_json(f"{base}/models", api_key=api_key)
    except (URLError, TimeoutError, ValueError, OSError):
        return []
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    for it in items:
        if isinstance(it, dict) and it.get("id"):
            out.append(LoadedModel(id=str(it["id"]), raw=it))
    return out


def pick_loaded_model(
    client: LiteLLMClient | None = None,
    *,
    prefer: str | None = None,
) -> LoadedModel | None:
    """Pick a currently-loaded model. Prefers *prefer* if loaded.

    If *prefer* (typically the user's configured model id) is loaded,
    returns it. Otherwise returns the first loaded model, or ``None``
    if nothing is loaded / reachable.
    """
    models = loaded_models(client)
    if not models:
        return None
    if prefer:
        # Match either bare id or any provider-prefixed form.
        bare = prefer.split("/", 1)[1] if "/" in prefer and prefer.split("/", 1)[0] in {
            "lm_studio", "openai", "anthropic", "ollama"
        } else prefer
        for m in models:
            if m.id == prefer or m.id == bare:
                return m
    return models[0]


def _first_int(d: dict[str, Any], *keys: str) -> int | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    return None


# ---------------------------------------------------------------------------
# Read / write the parameters cell of the settings notebook
# ---------------------------------------------------------------------------


def _parse_parameters_cell(source: str) -> dict[str, Any]:
    """Extract simple ``name = literal`` assignments from a parameters cell.

    Only top-level assignments to JSON-representable literals are honored
    (str, int, float, bool, None, list/dict of those). Anything else is
    ignored — this is by design, the parameters cell is meant to be
    declarative.
    """
    import ast

    out: dict[str, Any] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
        out[tgt.id] = value
    return out


def read_settings(path: Path | str) -> dict[str, Any]:
    """Read the parameters cell of a settings notebook.

    Returns an empty dict if the file is missing or has no parameters cell.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        nb = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    for cell in nb.get("cells", []):
        tags = cell.get("metadata", {}).get("tags", [])
        if cell.get("cell_type") == "code" and "parameters" in tags:
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            return _parse_parameters_cell(source)
    return {}


def write_settings(
    path: Path | str,
    settings: dict[str, Any],
    *,
    notes: str | None = None,
) -> Path:
    """Create or update a settings notebook with *settings* in its parameters cell.

    The notebook has three cells: a markdown header (with optional *notes*),
    the parameters cell (tagged ``parameters``), and a verification cell
    that prints the loaded settings when re-executed. Existing files are
    overwritten — this is meant to be the single canonical record per
    model, produced by the discovery skill.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = "# Model settings\n\nAuto-generated by `core.discover_model_settings`.\n"
    if notes:
        header += "\n" + notes.rstrip() + "\n"
    # Use Python ``repr`` (not JSON) so values like ``False``/``None`` are
    # Python-literal-eval-able when the notebook is re-read.
    params_src = "\n".join(f"{k} = {v!r}" for k, v in settings.items()) + "\n"
    verify_src = (
        "# Re-execute this notebook with Papermill to re-validate settings\n"
        "# against the live server.\n"
        "print({\n"
        + "".join(f"    {json.dumps(k)}: {k},\n" for k in settings)
        + "})\n"
    )
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": header,
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"tags": ["parameters"]},
                "outputs": [],
                "source": params_src,
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": verify_src,
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    p.write_text(json.dumps(nb, indent=1) + "\n")
    return p


__all__ = [
    "LoadedModel",
    "loaded_models",
    "model_slug",
    "pick_loaded_model",
    "read_settings",
    "settings_notebook_path",
    "write_settings",
]
