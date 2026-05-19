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
    kwargs: dict[str, Any] = dict(
        model=model,
        api_base=client.base_url,
        api_key=client.api_key,
        temperature=client.temperature,
        # dspy.LM defaults to 8 silent retries on failure. With a hung
        # thinking model that disconnects after the server's own timeout,
        # this looks like "the agent is stuck repeating the exact same
        # error forever". Fail fast — the agent loop can decide whether
        # to repair or surface to the user.
        num_retries=0,
        cache=False,
    )
    # Per the spec, the only enforced user-facing budgets are
    # ``max_autonomous_turns`` and lack-of-progress detection. Forward
    # ``max_tokens`` / ``timeout`` to LiteLLM ONLY when the caller asked
    # for them — otherwise let the provider use its own defaults (e.g.
    # LM Studio's ``-1`` = unlimited) so a thinking model isn't truncated
    # or cancelled mid-stream.
    if client.max_tokens is not None:
        kwargs["max_tokens"] = client.max_tokens
    if client.request_timeout is not None:
        kwargs["timeout"] = client.request_timeout
    # ``reasoning_effort`` handling.
    # LM Studio ignores this field on the OpenAI-compat endpoint and always
    # uses the per-model UI setting (bug-tracker issue #988), and sending an
    # unsupported value (e.g. "high" for Gemma) makes it emit a noisy WARN
    # and silently fall back. So: only forward the field for non-LM-Studio
    # providers. To control thinking on Gemma in LM Studio, use the
    # per-model Inference > Reasoning settings in the UI.
    if client.reasoning_effort is not None and client.provider != "lm_studio":
        kwargs["reasoning_effort"] = client.reasoning_effort
        kwargs["allowed_openai_params"] = ["reasoning_effort"]
    return dspy.LM(**kwargs)


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
