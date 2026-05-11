"""Security utilities for the agent-kernel runtime.

Public surface:

- ``redaction.redact(text)`` — pattern-based secret redaction returning
  text with detected secrets replaced by ``<REDACTED:KIND>`` placeholders.
- ``redaction.redact_payload(obj)`` — recursively redact strings in a
  JSON-compatible payload (dict/list/str/scalar).
- ``redaction.SENSITIVE_FIELD_NAMES`` — field names that are always
  redacted regardless of value (e.g. ``api_key``, ``password``, ``token``).

The JSONL store applies ``redact_payload`` to every event payload before
serialization so secrets never reach the durable ledger.
"""

from agent_kernel.security.redaction import (
    SENSITIVE_FIELD_NAMES,
    redact,
    redact_payload,
)

__all__ = ["SENSITIVE_FIELD_NAMES", "redact", "redact_payload"]
