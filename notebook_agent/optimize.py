"""Thin wrappers over DSPy optimizers (``MIPROv2`` and ``GEPA``).

These wrappers exist for two reasons:

1. ``import dspy`` is the only place the rest of the agent should touch the
   optimizer classes. Lazy-importing here means a user who never optimizes
   never pays for the optimizer dependencies (``gepa`` etc.).

2. The standard call shape (``optimizer.compile(program, trainset, …)``)
   accepts our :class:`notebook_agent.program.NotebookAgentProgram` directly,
   because that program is a real :class:`dspy.Module`. The wrappers add a
   little ergonomics — accepting any iterable of ``dspy.Example``-or-dict for
   the trainset, and providing a sensible default for ``auto="light"``.

Example::

    from notebook_agent import (
        NotebookAgentProgram, configure_dspy, optimize_with_mipro,
    )
    from notebook_agent.litellm_client import LiteLLMClient

    configure_dspy(LiteLLMClient())
    base = NotebookAgentProgram()
    compiled = optimize_with_mipro(
        base,
        trainset=[{"request": "count words in 'a b c'", "expected": 3}, ...],
        metric=my_metric,
        auto="light",
    )
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def _coerce_trainset(trainset: Iterable[Any]) -> list[Any]:
    """Normalize ``trainset`` into a list of ``dspy.Example`` objects.

    Items may already be ``dspy.Example`` (returned as-is), or ``dict`` (turned
    into examples with ``request`` as the input). Other shapes pass through.
    """
    import dspy  # type: ignore[import-untyped]

    out: list[Any] = []
    for item in trainset:
        if isinstance(item, dspy.Example):
            out.append(item)
        elif isinstance(item, dict):
            ex = dspy.Example(**item)
            # If 'request' is present, mark it as the input.
            if "request" in item:
                ex = ex.with_inputs("request")
            out.append(ex)
        else:
            out.append(item)
    return out


def optimize_with_mipro(
    program: Any,
    *,
    trainset: Iterable[Any],
    metric: Callable[..., float],
    auto: str | None = "light",
    num_threads: int | None = None,
    valset: Iterable[Any] | None = None,
    **mipro_kwargs: Any,
) -> Any:
    """Compile *program* with :class:`dspy.MIPROv2`.

    Returns the compiled program (a ``dspy.Module``) ready for inference.
    """
    from dspy.teleprompt import MIPROv2  # type: ignore[import-untyped]

    opt = MIPROv2(metric=metric, auto=auto, num_threads=num_threads, **mipro_kwargs)
    train = _coerce_trainset(trainset)
    val = _coerce_trainset(valset) if valset is not None else None
    kwargs: dict[str, Any] = {"trainset": train, "requires_permission_to_run": False}
    if val is not None:
        kwargs["valset"] = val
    return opt.compile(program, **kwargs)


def optimize_with_gepa(
    program: Any,
    *,
    trainset: Iterable[Any],
    metric: Callable[..., float],
    auto: str | None = "light",
    max_metric_calls: int | None = None,
    reflection_lm: Any | None = None,
    valset: Iterable[Any] | None = None,
    **gepa_kwargs: Any,
) -> Any:
    """Compile *program* with :class:`dspy.GEPA`.

    ``GEPA`` requires a reflection LM. If none is supplied, the current
    ``dspy.settings.lm`` is used so the call works out of the box when DSPy
    has already been configured.
    """
    import dspy  # type: ignore[import-untyped]
    from dspy.teleprompt import GEPA  # type: ignore[import-untyped]

    if reflection_lm is None:
        reflection_lm = getattr(dspy.settings, "lm", None)

    kwargs = dict(gepa_kwargs)
    if max_metric_calls is not None:
        kwargs.setdefault("max_metric_calls", max_metric_calls)
    else:
        kwargs.setdefault("auto", auto)
    kwargs.setdefault("reflection_lm", reflection_lm)

    opt = GEPA(metric=metric, **kwargs)
    train = _coerce_trainset(trainset)
    val = _coerce_trainset(valset) if valset is not None else None
    compile_kwargs: dict[str, Any] = {"trainset": train}
    if val is not None:
        compile_kwargs["valset"] = val
    return opt.compile(program, **compile_kwargs)


__all__ = ["optimize_with_gepa", "optimize_with_mipro"]
