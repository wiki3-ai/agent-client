"""LLM adapter layer.

Public surface:

- ``StructuredLLM`` — high-level adapter that takes a Pydantic
  ``response_model`` and returns a parsed instance, with retry-on-validation
  and per-call provenance + budget accounting.
- ``Provider`` — Protocol every concrete provider implements.
- ``FakeProvider`` — deterministic in-process provider used by CI
  integration tests.
- ``LMStudioProvider`` — OpenAI-compatible HTTP provider that targets a
  local LM Studio server (``http://localhost:1234/v1`` by default). Used
  for local development; tests against it are guarded by reachability
  probes and the ``llm`` pytest marker.
- ``LiteLLMProvider`` — optional adapter that delegates to ``litellm`` if
  installed (covers OpenAI, Anthropic, Azure, etc.). Imported lazily so
  the rest of the package doesn't take a hard dep.
"""

from agent_kernel.llm.adapter import (
    LLMCallError,
    LLMUsage,
    Provider,
    StructuredLLM,
)
from agent_kernel.llm.providers import FakeProvider, LMStudioProvider

__all__ = [
    "FakeProvider",
    "LLMCallError",
    "LLMUsage",
    "LMStudioProvider",
    "Provider",
    "StructuredLLM",
]
