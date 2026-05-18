"""Time helpers and stable ID generation."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime


def utcnow() -> datetime:
    """Return current UTC time with tzinfo set."""
    return datetime.now(UTC)


def iso_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return utcnow().isoformat()


def make_task_id() -> str:
    """Return a stable, short, random task ID."""
    return "task_" + secrets.token_hex(8)
