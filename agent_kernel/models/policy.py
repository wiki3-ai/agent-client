"""Policy profile model.

A policy profile parameterizes scheduling and admission decisions. Profiles
are loaded from JSON files or constructed in Python; the scheduler always
takes them as immutable typed inputs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent_kernel.models.budget import Budget


class ReservationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spawn_wall_ms_default: int = Field(default=300_000, ge=0)
    spawn_llm_usd_micro_default: int = Field(default=250_000, ge=0)
    parent_reserve_floor_ratio: float = Field(default=0.15, ge=0.0, le=1.0)


class Permissions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_network: bool = False
    allow_write: str = "workspace"  # "workspace" | "none" | "any"
    allow_shell: bool = False


class PolicyProfile(BaseModel):
    """Named, parameterized profile that defines budgets, quotas, and permissions."""

    model_config = ConfigDict(extra="forbid")

    name: str
    max_spawn_depth: int = Field(default=3, ge=0)
    max_children_per_task: int = Field(default=6, ge=0)
    kernel_slots_total: int = Field(default=4, ge=1)
    queue_limit: int = Field(default=64, ge=1)
    reservation_policy: ReservationPolicy = Field(default_factory=ReservationPolicy)
    budgets: Budget = Field(default_factory=Budget)
    permissions: Permissions = Field(default_factory=Permissions)


def local_dev_profile() -> PolicyProfile:
    """Default profile for local development."""
    return PolicyProfile(
        name="local-dev",
        max_spawn_depth=3,
        max_children_per_task=6,
        kernel_slots_total=4,
        queue_limit=64,
        budgets=Budget(
            wall_ms=1_800_000,
            cpu_ms=600_000,
            llm_usd_micro=2_000_000,
            llm_input_tokens=250_000,
            llm_output_tokens=100_000,
            spawn_count=12,
        ),
        permissions=Permissions(allow_network=True, allow_write="workspace", allow_shell=False),
    )


def research_profile() -> PolicyProfile:
    """Larger default profile for research runs."""
    p = local_dev_profile()
    return p.model_copy(
        update={
            "name": "research",
            "max_spawn_depth": 5,
            "max_children_per_task": 12,
            "kernel_slots_total": 8,
            "queue_limit": 256,
            "budgets": Budget(
                wall_ms=7_200_000,
                cpu_ms=3_600_000,
                llm_usd_micro=20_000_000,
                llm_input_tokens=2_000_000,
                llm_output_tokens=800_000,
                spawn_count=64,
            ),
        }
    )


DEFAULT_PROFILES: dict[str, PolicyProfile] = {
    "local-dev": local_dev_profile(),
    "research": research_profile(),
}
