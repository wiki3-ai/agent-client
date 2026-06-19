"""LLM adapter layer.

Public surface:

- ``StructuredLLM`` — high-level adapter that takes a Pydantic
  ``response_model`` and returns a parsed instance, with retry-on-validation
  and per-call provenance + budget accounting.
- ``Provider`` — Protocol every concrete provider implements.
- ``LiteLLMProvider`` — thin wrapper around :func:`litellm.completion`.
  All real LLM logic lives in LiteLLM.
- ``FakeProvider`` — deterministic preset that scripts LiteLLM's
  canonical ``mock_response`` kwarg
  (see https://docs.litellm.ai/docs/completion/mock_requests).
- ``LMStudioProvider`` — preset that routes through LiteLLM's built-in
  ``lm_studio/<model>`` provider, auto-detecting the loaded model from
  ``{base_url}/models`` when one isn't given.
"""

from agent_kernel.llm.adapter import (
    LLMCallError,
    LLMUsage,
    Provider,
    StructuredLLM,
)
from agent_kernel.llm.providers import FakeProvider, LiteLLMProvider, LMStudioProvider

__all__ = [
    "FakeProvider",
    "LLMCallError",
    "LLMUsage",
    "LMStudioProvider",
    "LiteLLMProvider",
    "Provider",
    "StructuredLLM",
]
