"""``agent-kernel`` control kernel.

This kernel is a thin subclass of ``IPythonKernel`` that intercepts cells
starting with ``%agent`` and dispatches them through ``agent_kernel.magics``.
Cells that don't begin with ``%agent`` execute as normal Python.

The kernel maintains a single ``AgentKernel`` facade per kernel process,
bound to the workspace identified by the ``AGENT_KERNEL_WORKSPACE`` env
var (default: the kernel process's current working directory).
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from ipykernel.ipkernel import IPythonKernel

from agent_kernel import __version__
from agent_kernel.api import AgentKernel
from agent_kernel.magics import MagicError, dispatch, is_agent_magic, workspace_from_env


class AgentControlKernel(IPythonKernel):
    """Jupyter kernel that exposes the agent-kernel orchestration surface."""

    implementation = "agent-kernel"
    implementation_version = __version__
    language_info: ClassVar[dict[str, str]] = {
        "name": "python",
        "mimetype": "text/x-python",
        "file_extension": ".py",
    }
    banner = f"agent-kernel {__version__} — type %agent help"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._agent: AgentKernel | None = None

    @property
    def agent(self) -> AgentKernel:
        if self._agent is None:
            self._agent = AgentKernel(workspace_from_env())
        return self._agent

    async def do_execute(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: dict[str, Any] | None = None,
        allow_stdin: bool = False,
        *,
        cell_id: str | None = None,
    ) -> dict[str, Any]:
        # Process leading agent-magic lines; the rest (if any) falls through
        # to normal Python execution.
        lines = code.splitlines()
        magic_lines: list[str] = []
        remainder: list[str] = []
        for i, line in enumerate(lines):
            if is_agent_magic(line):
                magic_lines.append(line)
            else:
                remainder = lines[i:]
                break

        for line in magic_lines:
            try:
                result = dispatch(line, self.agent)
                if not silent:
                    self._stream("stdout", result + "\n")
            except MagicError as exc:
                payload = json.dumps({"ok": False, "error": str(exc)})
                if not silent:
                    self._stream("stderr", payload + "\n")
                return {
                    "status": "error",
                    "execution_count": self.execution_count,
                    "ename": "MagicError",
                    "evalue": str(exc),
                    "traceback": [str(exc)],
                }

        if remainder:
            return await super().do_execute(
                "\n".join(remainder),
                silent,
                store_history,
                user_expressions,
                allow_stdin,
                cell_id=cell_id,
            )
        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }

    # ------------------------------------------------------------ stream helper

    def _stream(self, name: str, text: str) -> None:
        self.send_response(self.iopub_socket, "stream", {"name": name, "text": text})


def launch() -> None:  # pragma: no cover — invoked by kernelspec only
    """Entry point for the kernelspec ``argv``."""
    from ipykernel.kernelapp import IPKernelApp

    IPKernelApp.launch_instance(kernel_class=AgentControlKernel)


if __name__ == "__main__":  # pragma: no cover
    launch()
