"""Public Python API for agent-kernel.

This module is the small, typed surface contributors and downstream tools
program against. It is intentionally minimal; everything is delegated to
the scheduler and runtime services.

Usage::

    from agent_kernel.api import AgentKernel
    ak = AgentKernel(workspace="./my-workspace")
    task = ak.create_task(notebook_path="tasks/parent.ipynb")
    result = ak.run_task(task.task_id)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent_kernel.models.budget import Budget
from agent_kernel.models.event import ProvenanceEvent
from agent_kernel.models.policy import DEFAULT_PROFILES, PolicyProfile
from agent_kernel.models.task import SpawnSpec, TaskSpec
from agent_kernel.runtime.scheduler import Scheduler
from agent_kernel.runtime.spawn_manager import SpawnManager, SpawnResult
from agent_kernel.storage import WorkspaceLayout

__all__ = [
    "AgentKernel",
    "Budget",
    "PolicyProfile",
    "ProvenanceEvent",
    "SpawnResult",
    "SpawnSpec",
    "TaskSpec",
    "create_task",
    "get_task",
    "list_events",
    "run_task",
]


class AgentKernel:
    """High-level facade over a single workspace + scheduler."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        policy_profile: str | PolicyProfile = "local-dev",
    ) -> None:
        self.workspace = WorkspaceLayout(workspace)
        profile = (
            policy_profile
            if isinstance(policy_profile, PolicyProfile)
            else DEFAULT_PROFILES[policy_profile]
        )
        self.scheduler = Scheduler(self.workspace, profile=profile)
        self.spawn_manager = SpawnManager(self.scheduler)

    # --- task lifecycle ---------------------------------------------------

    def create_task(self, **kwargs: object) -> TaskSpec:
        return self.scheduler.create_task(**kwargs)  # type: ignore[arg-type]

    def get_task(self, task_id: str) -> TaskSpec | None:
        return self.scheduler.get_task(task_id)

    def list_events(self, task_id: str | None = None) -> list[ProvenanceEvent]:
        return self.scheduler.list_events(task_id)

    def cancel(self, task_id: str) -> None:
        self.scheduler.cancel(task_id)

    def run_task(self, task_id: str) -> TaskSpec:
        """Synchronous wrapper around ``Scheduler.run_task``.

        Tolerates being called from inside a running event loop (e.g. from a
        notebook cell in an ipykernel) by dispatching to a worker thread
        with its own loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.scheduler.run_task(task_id))

        # We're already inside an event loop — run on a side thread.
        import concurrent.futures

        def _run() -> TaskSpec:
            return asyncio.run(self.scheduler.run_task(task_id))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run).result()

    def spawn_child_task(self, parent_task_id: str, spec: SpawnSpec) -> SpawnResult:
        """Spawn a child task from ``parent_task_id``. Caller must run it."""
        return self.spawn_manager.spawn(parent_task_id, spec)


# --- module-level convenience (M4 spec wording) ----------------------------


def create_task(workspace: str | Path, **kwargs: object) -> TaskSpec:
    return AgentKernel(workspace).create_task(**kwargs)


def run_task(workspace: str | Path, task_id: str) -> TaskSpec:
    return AgentKernel(workspace).run_task(task_id)


def get_task(workspace: str | Path, task_id: str) -> TaskSpec | None:
    return AgentKernel(workspace).get_task(task_id)


def list_events(workspace: str | Path, task_id: str | None = None) -> list[ProvenanceEvent]:
    return AgentKernel(workspace).list_events(task_id)
