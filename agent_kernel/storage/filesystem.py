"""Workspace layout: filesystem conventions under ``<workspace>/.agent_kernel/``.

The source notebook tree stays visible at the workspace root; all
operational state is hidden under ``.agent_kernel/``.
"""

from __future__ import annotations

from pathlib import Path


class WorkspaceLayout:
    """Filesystem layout for a single agent-kernel workspace."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    # --- top-level paths ---------------------------------------------------

    @property
    def state_root(self) -> Path:
        return self.root / ".agent_kernel"

    @property
    def config_path(self) -> Path:
        return self.state_root / "config.json"

    @property
    def tasks_dir(self) -> Path:
        return self.state_root / "tasks"

    @property
    def runs_dir(self) -> Path:
        return self.state_root / "runs"

    @property
    def events_dir(self) -> Path:
        return self.state_root / "events"

    @property
    def artifacts_dir(self) -> Path:
        return self.state_root / "artifacts"

    @property
    def notebooks_dir(self) -> Path:
        return self.root / "notebooks"

    @property
    def spawns_dir(self) -> Path:
        return self.notebooks_dir / "spawns"

    # --- task/run helpers --------------------------------------------------

    def task_state_path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def artifact_path(self, artifact_id: str) -> Path:
        return self.artifacts_dir / f"{artifact_id}.json"

    # --- setup -------------------------------------------------------------

    def ensure(self) -> None:
        """Create all known subdirectories. Idempotent."""
        for d in [
            self.state_root,
            self.tasks_dir,
            self.runs_dir,
            self.events_dir,
            self.artifacts_dir,
            self.notebooks_dir,
            self.spawns_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
