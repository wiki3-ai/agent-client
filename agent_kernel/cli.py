"""CLI entry points for agent-kernel.

Subcommands are wired in incrementally per milestone:
- ``agent-kernel run``     — Milestone 2 (notebook runner)
- ``agent-kernel scaffold``— Milestone 3 (template materializer)
- ``agent-kernel inspect`` — Milestone 4 (task lifecycle)
- ``agent-kernel install`` — Milestone 6 (kernelspec install)
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from agent_kernel import __version__


@click.group(help="agent-kernel command-line interface.")
@click.version_option(version=__version__, prog_name="agent-kernel")
def main() -> None:
    """Root command group."""


@main.command(help="Print version and exit.")
def version() -> None:
    click.echo(__version__)


@main.command(help="Execute a notebook with provenance emission.")
@click.argument("notebook", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--workspace",
    "-w",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path.cwd(),
    show_default=False,
    help="Workspace root containing .agent_kernel/ state (defaults to cwd).",
)
@click.option("--kernel", "-k", default="python3", show_default=True, help="Jupyter kernel name.")
@click.option(
    "--timeout", type=int, default=60, show_default=True, help="Per-cell timeout seconds."
)
@click.option(
    "--task-id",
    default=None,
    help="Reuse an existing task id; otherwise a new one is generated.",
)
def run(notebook: Path, workspace: Path, kernel: str, timeout: int, task_id: str | None) -> None:
    """Run a notebook and emit a JSONL provenance trace."""
    from agent_kernel.runtime.notebook_runner import NotebookRunFailed, NotebookRunner
    from agent_kernel.storage import JSONLEventStore, WorkspaceLayout
    from agent_kernel.util import new_id

    ws = WorkspaceLayout(workspace)
    ws.ensure()
    events = JSONLEventStore(ws.events_dir, fsync=True)
    runner = NotebookRunner(events, ws.runs_dir)
    tid = task_id or new_id("task")
    try:
        result = runner.run(notebook, task_id=tid, kernel_name=kernel, timeout=timeout)
    except NotebookRunFailed as exc:
        click.echo(f"FAILED run_id={exc.run_id} reason={exc.reason}", err=True)
        sys.exit(1)
    click.echo(
        f"OK task_id={result.task_id} run_id={result.run_id} "
        f"executed_notebook={result.executed_notebook_path}"
    )


@main.command(help="Install the agent-kernel kernelspec for Jupyter.")
@click.option("--user", is_flag=True, default=False, help="Install for the current user.")
@click.option("--prefix", default=None, help="Install under this prefix.")
@click.option("--name", default="agent-kernel", show_default=True, help="Kernelspec name.")
def install(user: bool, prefix: str | None, name: str) -> None:
    from agent_kernel.install import install_kernelspec

    dest = install_kernelspec(name=name, user=user, prefix=prefix)
    click.echo(f"Installed kernelspec {name!r} -> {dest}")


if __name__ == "__main__":  # pragma: no cover
    main()
