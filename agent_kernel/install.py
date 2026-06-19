"""Kernelspec install helper.

Usage::

    python -m agent_kernel install [--user] [--prefix PREFIX] [--name agent-kernel]

Writes a ``kernel.json`` to the appropriate Jupyter kernels directory so
that JupyterLab / Jupyter Server can launch the agent-kernel control kernel.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import click
from jupyter_client.kernelspec import KernelSpecManager


def make_kernelspec_dict(display_name: str = "agent-kernel") -> dict:
    return {
        "argv": [sys.executable, "-m", "agent_kernel.kernel", "-f", "{connection_file}"],
        "display_name": display_name,
        "language": "python",
        "metadata": {"debugger": False},
    }


def install_kernelspec(
    *,
    name: str = "agent-kernel",
    display_name: str = "agent-kernel",
    user: bool = True,
    prefix: str | None = None,
) -> Path:
    """Install the agent-kernel kernelspec and return the installed directory."""
    spec = make_kernelspec_dict(display_name)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "kernel.json").write_text(json.dumps(spec, indent=2))
        manager = KernelSpecManager()
        dest = manager.install_kernel_spec(str(td_path), kernel_name=name, user=user, prefix=prefix)
    return Path(dest)


def uninstall_kernelspec(name: str = "agent-kernel") -> None:
    manager = KernelSpecManager()
    try:
        spec = manager.get_kernel_spec(name)
    except Exception:
        return
    shutil.rmtree(spec.resource_dir, ignore_errors=True)


@click.command(help="Install the agent-kernel kernelspec for Jupyter.")
@click.option("--user", is_flag=True, default=False, help="Install for the current user.")
@click.option("--prefix", default=None, help="Install under this prefix.")
@click.option("--name", default="agent-kernel", show_default=True, help="Kernelspec name.")
@click.option("--display-name", default="agent-kernel", show_default=True, help="Display name.")
def main(user: bool, prefix: str | None, name: str, display_name: str) -> None:
    dest = install_kernelspec(name=name, display_name=display_name, user=user, prefix=prefix)
    click.echo(f"Installed kernelspec {name!r} -> {dest}")


if __name__ == "__main__":  # pragma: no cover
    main()
