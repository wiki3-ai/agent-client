"""``ExtensionApp`` wiring agent-kernel into Jupyter Server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jupyter_server.extension.application import ExtensionApp
from traitlets import Unicode

from agent_kernel.api import AgentKernel
from agent_kernel_server.handlers import (
    HealthHandler,
    TaskEventsHandler,
    TaskHandler,
    TaskRunHandler,
    TasksHandler,
)


class AgentKernelExtension(ExtensionApp):
    """Jupyter Server extension exposing the agent-kernel HTTP API."""

    name = "agent_kernel_server"
    extension_url = "/agent-kernel"
    load_other_extensions = False

    workspace = Unicode(
        "",
        config=True,
        help=(
            "Workspace directory for agent-kernel state. Defaults to "
            "$AGENT_KERNEL_WORKSPACE or the server's root dir."
        ),
    )

    def initialize_settings(self) -> None:  # type: ignore[override]
        ws_path = (
            self.workspace or os.environ.get("AGENT_KERNEL_WORKSPACE") or self.serverapp.root_dir  # type: ignore[union-attr]
        )
        agent_kernel = AgentKernel(Path(ws_path))
        self.settings["agent_kernel"] = agent_kernel
        self.log.info("agent-kernel workspace: %s", ws_path)

    def initialize_handlers(self) -> None:  # type: ignore[override]
        # Use a non-greedy regex so trailing path segments are matched cleanly.
        base = r"/api/agent-kernel"
        handlers: list[tuple[str, type[Any], dict | None]] = [
            (f"{base}/health", HealthHandler, None),
            (f"{base}/tasks", TasksHandler, None),
            (rf"{base}/tasks/(?P<task_id>[^/]+)", TaskHandler, None),
            (rf"{base}/tasks/(?P<task_id>[^/]+)/run", TaskRunHandler, None),
            (rf"{base}/tasks/(?P<task_id>[^/]+)/events", TaskEventsHandler, None),
        ]
        # Tornado expects 2- or 3-tuples
        self.handlers.extend(  # type: ignore[attr-defined]
            [(pat, cls) if extra is None else (pat, cls, extra) for pat, cls, extra in handlers]
        )
