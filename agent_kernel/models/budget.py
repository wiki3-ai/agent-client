"""Budget and quota models.

Budgets are consumable numeric allowances (wall-clock, CPU, LLM spend,
tokens, bytes written, spawn count). Quotas are concurrency ceilings.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Budget(BaseModel):
    """Consumable budget for a task.

    All fields are non-negative integers. ``llm_usd_micro`` is USD * 1e6 to
    avoid floating-point accounting drift.
    """

    model_config = ConfigDict(extra="forbid", strict=False)

    wall_ms: int = Field(default=0, ge=0)
    cpu_ms: int = Field(default=0, ge=0)
    llm_usd_micro: int = Field(default=0, ge=0)
    llm_input_tokens: int = Field(default=0, ge=0)
    llm_output_tokens: int = Field(default=0, ge=0)
    bytes_written: int = Field(default=0, ge=0)
    spawn_count: int = Field(default=0, ge=0)

    def can_cover(self, other: Budget) -> bool:
        """Return ``True`` iff every field of ``self`` is >= ``other``."""
        return all(getattr(self, f) >= getattr(other, f) for f in self.__class__.model_fields)

    def subtract(self, other: Budget) -> Budget:
        """Return a new Budget = self - other. Raises ValueError on underflow."""
        result = {}
        for f in self.__class__.model_fields:
            v = getattr(self, f) - getattr(other, f)
            if v < 0:
                raise ValueError(
                    f"Budget underflow on field {f}: {getattr(self, f)} - {getattr(other, f)}"
                )
            result[f] = v
        return Budget(**result)

    def add(self, other: Budget) -> Budget:
        return Budget(
            **{f: getattr(self, f) + getattr(other, f) for f in self.__class__.model_fields}
        )


class QuotaSnapshot(BaseModel):
    """Concurrency-slot snapshot at a point in time."""

    model_config = ConfigDict(extra="forbid")

    kernel_slots_used: int = Field(default=0, ge=0)
    kernel_slots_total: int = Field(default=0, ge=0)
    running_tasks: int = Field(default=0, ge=0)
    queued_tasks: int = Field(default=0, ge=0)
