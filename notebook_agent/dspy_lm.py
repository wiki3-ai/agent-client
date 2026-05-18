"""Bridge :class:`notebook_agent.litellm_client.LiteLLMClient` to ``dspy.LM``.

DSPy expects every LM call to go through ``dspy.settings.lm`` (a ``dspy.LM``).
We already configure LiteLLM with our own env vars and a fake provider for
tests, so this module's job is to translate those settings into a DSPy LM
and configure DSPy. The same client object that drives logging continues to
own the call log; DSPy itself drives the prompt/response formatting.

For ``provider="fake"`` we use :class:`dspy.utils.dummies.DummyLM` so tests
remain deterministic without touching LiteLLM's network paths.
"""

from __future__ import annotations

import contextlib
from typing import Any

from .litellm_client import LiteLLMClient


def build_dspy_lm(client: LiteLLMClient, *, fake_answers: list[dict[str, Any]] | None = None) -> Any:
    """Return a ``dspy.LM`` (or ``DummyLM``) configured from *client*.

    For the fake provider, the caller may pass ``fake_answers`` — a list of
    dicts mapping declared output-field names to their canned values, exactly
    as :class:`dspy.utils.dummies.DummyLM` consumes them. If omitted, the
    client's own ``fake_answers`` / ``fake_response`` is used.
    """
    import dspy  # type: ignore[import-untyped]

    if client.provider == "fake":
        from dspy.utils.dummies import DummyLM  # type: ignore[import-untyped]

        if fake_answers is None:
            fake_answers = client.fake_answers
        if fake_answers is None:
            fake_answers = [{"answer": client.fake_response or ""}]
        return DummyLM(fake_answers)

    # Map our LM-Studio / OpenAI-compatible config onto a dspy.LM. DSPy uses
    # the LiteLLM-style ``<provider>/<model>`` syntax. We pass api_base/api_key
    # through directly so LM Studio works out of the box.
    model = client.model
    if "/" not in model:
        model = f"openai/{model}"
    return dspy.LM(
        model=model,
        api_base=client.base_url,
        api_key=client.api_key,
        temperature=client.temperature,
        max_tokens=client.max_tokens,
        cache=False,
    )


def configure_dspy(client: LiteLLMClient, *, fake_answers: list[dict[str, Any]] | None = None) -> Any:
    """Configure ``dspy.settings.lm`` from *client* and return the LM."""
    import dspy  # type: ignore[import-untyped]

    lm = build_dspy_lm(client, fake_answers=fake_answers)
    dspy.configure(lm=lm)
    return lm


@contextlib.contextmanager
def using_client(client: LiteLLMClient, *, fake_answers: list[dict[str, Any]] | None = None):
    """Context-manager flavor of :func:`configure_dspy`.

    Restores the previous ``dspy.settings.lm`` on exit, so callers that
    already configured DSPy globally are not disturbed.
    """
    import dspy  # type: ignore[import-untyped]

    prev = getattr(dspy.settings, "lm", None)
    lm = configure_dspy(client, fake_answers=fake_answers)
    try:
        yield lm
    finally:
        try:
            dspy.configure(lm=prev)
        except Exception:
            pass


__all__ = ["build_dspy_lm", "configure_dspy", "using_client"]
