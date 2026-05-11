"""Shared utilities: timestamps, identifiers, and event-trace state reconstruction."""

from __future__ import annotations

import secrets
import string
from datetime import UTC, datetime

# Base32 Crockford alphabet without ambiguous characters.
_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def now_iso() -> str:
    """Return current time as RFC 3339 in UTC, millisecond precision."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_id(prefix: str, length: int = 24) -> str:
    """Return a sortable-ish opaque ID with the given prefix.

    Not a true ULID; sufficient for MVP uniqueness and human-readable prefixes.
    """
    alphabet = string.ascii_uppercase + string.digits
    body = "".join(secrets.choice(alphabet) for _ in range(length))
    return f"{prefix}_{body}"
