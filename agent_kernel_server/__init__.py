"""agent_kernel_server: Jupyter Server extension for agent-kernel."""

__version__ = "0.0.1"


def _jupyter_server_extension_points() -> list[dict]:
    """Jupyter Server extension discovery hook."""
    # Import lazily inside the function so static analyzers / loaders
    # don't import the heavy ExtensionApp at package-import time.
    from agent_kernel_server.app import AgentKernelExtension

    return [{"module": "agent_kernel_server", "app": AgentKernelExtension}]


def _load_jupyter_server_extension(server_app: object) -> None:
    """Legacy load hook (calls into ExtensionApp)."""
    from agent_kernel_server.app import AgentKernelExtension

    ext = AgentKernelExtension()
    ext._link_jupyter_server_extension(server_app)  # type: ignore[attr-defined]
    ext.initialize()


# Re-export
from agent_kernel_server.app import AgentKernelExtension  # noqa: E402

__all__ = ["AgentKernelExtension", "__version__"]
