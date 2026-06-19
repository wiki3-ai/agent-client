"""Storage primitives for agent-kernel.

All durable I/O lives here. The MVP storage stack is:
- ``jsonl_store``: append-only provenance event log with daily rotation
- ``state_store``: atomic JSON state for tasks and artifacts
- ``filesystem``: workspace layout helpers
"""

from agent_kernel.storage.filesystem import WorkspaceLayout
from agent_kernel.storage.jsonl_store import JSONLEventStore
from agent_kernel.storage.state_store import AtomicJSONStore

__all__ = ["AtomicJSONStore", "JSONLEventStore", "WorkspaceLayout"]
