"""agent_kernel_server: Jupyter Server extension for agent-kernel.

Implemented in Milestone 8. This module currently exposes the
extension-discovery entry points only.
"""

__version__ = "0.0.1"


def _jupyter_server_extension_points() -> list[dict]:
    """Return the Jupyter Server extension entry point.

    Stubbed for M0; the actual ``ExtensionApp`` is implemented in M8.
    """
    return [{"module": "agent_kernel_server"}]
