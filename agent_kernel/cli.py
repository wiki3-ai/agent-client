"""CLI entry points for agent-kernel.

Subcommands are wired in incrementally per milestone:
- ``agent-kernel run``     — Milestone 2 (notebook runner)
- ``agent-kernel scaffold``— Milestone 3 (template materializer)
- ``agent-kernel inspect`` — Milestone 4 (task lifecycle)
- ``agent-kernel install`` — Milestone 6 (kernelspec install)
"""

from __future__ import annotations

import click

from agent_kernel import __version__


@click.group(help="agent-kernel command-line interface.")
@click.version_option(version=__version__, prog_name="agent-kernel")
def main() -> None:
    """Root command group."""


@main.command(help="Print version and exit.")
def version() -> None:
    click.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    main()
