"""M0 integration gate: package import + CLI invocation smoke.

Proves the build/install/test loop works end-to-end on a clean checkout.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.integration
def test_packages_import_and_have_version() -> None:
    import agent_kernel
    import agent_kernel_server

    assert isinstance(agent_kernel.__version__, str)
    assert agent_kernel.__version__ == agent_kernel_server.__version__
    assert agent_kernel.__version__.count(".") >= 2


@pytest.mark.integration
def test_cli_version_subprocess() -> None:
    """Invoke the installed CLI as a subprocess and assert version output."""
    result = subprocess.run(
        [sys.executable, "-m", "agent_kernel.cli", "version"],
        check=True,
        capture_output=True,
        text=True,
    )
    import agent_kernel

    assert result.stdout.strip() == agent_kernel.__version__


@pytest.mark.integration
def test_jupyter_server_extension_discovery() -> None:
    import agent_kernel_server

    points = agent_kernel_server._jupyter_server_extension_points()
    assert points == [{"module": "agent_kernel_server"}]
