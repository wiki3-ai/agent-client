"""Notebook initialization & DSPy-optimizable parameters.

This module gives a *user-facing* notebook (e.g. ``examples/first.ipynb``)
a single, declarative way to:

1. Set its Papermill **parameters** cell (model, base_url, max_tokens,
   temperature, max_autonomous_turns, runs_root, …) with sane defaults that
   come from environment variables.
2. Apply those parameters: build a :class:`LiteLLMClient`, configure DSPy's
   global LM (``dspy.settings.lm``), and stash notebook-wide knobs
   (``max_autonomous_turns``, ``runs_root``, ``skill_dirs``) that the
   ``%task`` / ``%%task`` magics will pick up.

The same parameter set is the **DSPy GEPA hyperparameter search space**.
GEPA can read :func:`notebook_parameters` to discover the optimizable
fields, mutate them, re-execute the user notebook via Papermill, and score
each trial against the trajectory captured under ``runs/`` (events, task
manifests, executed notebooks). Keep the parameter names stable and
documented.

Typical notebook usage::

    # cell tagged "parameters" (Papermill contract)
    provider = None
    model = None
    base_url = None
    api_key = None
    max_tokens = None      # → env NOTEBOOK_AGENT_MAX_TOKENS or 16384
    temperature = None     # → env NOTEBOOK_AGENT_TEMPERATURE or 0.0
    max_autonomous_turns = 6
    runs_root = "runs"
    skill_dirs = []

    # next cell — applies the parameters
    import notebook_agent as na
    na.init_notebook(
        provider=provider, model=model, base_url=base_url, api_key=api_key,
        max_tokens=max_tokens, temperature=temperature,
        max_autonomous_turns=max_autonomous_turns,
        runs_root=runs_root, skill_dirs=skill_dirs,
    )

After ``init_notebook(...)``, every ``%task`` in the notebook uses these
defaults. Per-call magic flags (``--max-tokens N`` / ``--temperature F``)
still override on a single call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .dspy_lm import configure_dspy
from .litellm_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_TEMPERATURE,
    LiteLLMClient,
)


@dataclass
class NotebookConfig:
    """Notebook-wide defaults consulted by the ``%task`` magics."""

    client: LiteLLMClient | None = None
    max_autonomous_turns: int = 6
    runs_root: str = "runs"
    skill_dirs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # The client doesn't serialize cleanly via asdict; replace it with
        # the public knobs that actually went into building it.
        d["client"] = self.client.to_dict() if self.client is not None else None
        return d


# Module-level singleton. ``init_notebook`` updates it; the magics read it.
_NB_CONFIG: NotebookConfig = NotebookConfig()


def get_notebook_config() -> NotebookConfig:
    """Return the current notebook-wide configuration."""
    return _NB_CONFIG


def init_notebook(
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    request_timeout: float | None = None,
    reasoning_effort: str | None = ...,  # type: ignore[assignment]
    max_autonomous_turns: int = 6,
    runs_root: str | Path = "runs",
    skill_dirs: list[str | Path] | None = None,
) -> LiteLLMClient:
    """Apply notebook parameters → configure DSPy → stash notebook config.

    All arguments are optional; ``None`` means *fall back to env-var defaults*
    (see :class:`LiteLLMClient`). Returns the configured
    :class:`LiteLLMClient` so the cell renders something useful when the
    notebook is read by a human.
    """
    global _NB_CONFIG

    client = LiteLLMClient(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        request_timeout=request_timeout,
        **({} if reasoning_effort is ... else {"reasoning_effort": reasoning_effort}),
    )
    configure_dspy(client)

    _NB_CONFIG = NotebookConfig(
        client=client,
        max_autonomous_turns=int(max_autonomous_turns),
        runs_root=str(runs_root),
        skill_dirs=[str(p) for p in (skill_dirs or [])],
    )
    return client


def notebook_parameters() -> list[dict[str, Any]]:
    """Schema of optimizable notebook parameters (DSPy GEPA search space).

    Each entry describes one Papermill parameter the user notebook is
    expected to expose. GEPA-style optimizers read this to know which
    hyperparameters they may mutate across re-executions of the notebook.
    """
    return [
        {
            "name": "max_tokens",
            "type": "int",
            "default": DEFAULT_MAX_TOKENS,
            "min": 1024,
            "max": 65536,
            "description": "Per-call max output tokens. Thinking models need ≥16k.",
        },
        {
            "name": "temperature",
            "type": "float",
            "default": DEFAULT_TEMPERATURE,
            "min": 0.0,
            "max": 2.0,
            "description": "LM sampling temperature.",
        },
        {
            "name": "request_timeout",
            "type": "float",
            "default": DEFAULT_REQUEST_TIMEOUT,
            "min": 30.0,
            "max": 1800.0,
            "description": "Hard wall-clock cap on a single LM HTTP request (seconds).",
        },
        {
            "name": "reasoning_effort",
            "type": "str",
            "default": "low",
            "choices": ["low", "medium", "high", None],
            "description": "Cap on thinking-model reasoning tokens. 'low' avoids 10k-token scratchpad loops.",
        },
        {
            "name": "max_autonomous_turns",
            "type": "int",
            "default": 6,
            "min": 1,
            "max": 32,
            "description": "Max LLM-driven steps before the agent reports back.",
        },
        {
            "name": "model",
            "type": "str",
            "default": None,
            "description": "LiteLLM model id (e.g. lm_studio/google/gemma-4-31b).",
        },
        {
            "name": "provider",
            "type": "str",
            "default": None,
            "description": "LiteLLM provider prefix (lm_studio, openai, fake, ...).",
        },
    ]


__all__ = [
    "NotebookConfig",
    "get_notebook_config",
    "init_notebook",
    "notebook_parameters",
]
