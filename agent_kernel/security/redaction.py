"""Pattern-based secret redaction for the JSONL ledger.

This is a defense-in-depth measure. The primary controls are still:

- Don't put secrets in code, notebook source, or task parameters.
- Source API keys from environment variables or the server's secret
  store, not from request bodies or notebook metadata.

When secrets do leak — for example an LLM provider key copied into a
prompt — the JSONL writer scrubs them before persistence so the durable
ledger remains safe to share, archive, and replay.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

# Tuples of (kind, compiled_pattern). Each pattern's *entire match* is
# replaced; group(0) is the secret. Order matters — more specific
# patterns come first so the generic high-entropy fallback doesn't strip
# meaningful prefixes.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Anthropic keys: sk-ant-api03-... (match BEFORE generic openai so it wins).
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    # OpenAI-style keys: sk-... or sk-proj-... but NOT sk-ant-...
    ("openai_key", re.compile(r"sk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{20,}")),
    # GitHub PAT (classic and fine-grained)
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    # AWS access key id
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    # Generic Bearer tokens in Authorization headers
    (
        "bearer_token",
        re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),
    ),
    # JWTs (three base64url segments)
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
    ),
    # Private key blocks
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
]


# Field names whose values are *always* redacted, even if the value's
# content doesn't match any pattern.
SENSITIVE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "auth_token",
        "password",
        "passwd",
        "secret",
        "secret_key",
        "token",
        "access_token",
        "refresh_token",
        "private_key",
        "credentials",
        "openai_api_key",
        "anthropic_api_key",
        "github_token",
        "aws_secret_access_key",
        "session_token",
    }
)

_REDACTED = "<REDACTED:{kind}>"


def redact(
    text: str, *, extra_patterns: Iterable[tuple[str, re.Pattern[str]]] | None = None
) -> str:
    """Return ``text`` with detected secrets replaced.

    ``extra_patterns`` may extend the default pattern list (callers should
    register provider-specific patterns this way rather than monkey-patching).
    """
    out = text
    patterns = list(_PATTERNS)
    if extra_patterns:
        patterns.extend(extra_patterns)
    for kind, pat in patterns:
        out = pat.sub(_REDACTED.format(kind=kind), out)
    return out


def redact_payload(obj: Any) -> Any:
    """Recursively redact strings inside a JSON-compatible payload.

    - Dict values whose key is in :data:`SENSITIVE_FIELD_NAMES` (case-
      insensitive) are replaced entirely with ``<REDACTED:field_name>``.
    - Dict values otherwise are recursed into.
    - Lists / tuples are recursed element-wise.
    - Strings are passed through :func:`redact`.
    - All other types are returned unchanged.

    The returned object is fresh (not a mutation of ``obj``).
    """
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in SENSITIVE_FIELD_NAMES:
                out[k] = _REDACTED.format(kind=k.lower())
            else:
                out[k] = redact_payload(v)
        return out
    if isinstance(obj, list):
        return [redact_payload(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(redact_payload(v) for v in obj)
    if isinstance(obj, str):
        return redact(obj)
    return obj
