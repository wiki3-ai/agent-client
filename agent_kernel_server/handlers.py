"""Tornado handlers exposing the agent-kernel HTTP API.

Authentication is enforced by Jupyter Server's ``@authenticated`` decorator;
authorization respects the configured ``Authorizer``. All handlers serialize
``TaskSpec`` and ``ProvenanceEvent`` Pydantic models through their canonical
``model_dump(mode="json")`` form so the wire schema matches the JSONL
schema bit-for-bit.
"""

from __future__ import annotations

import json
from typing import Any

from jupyter_server.auth import authorized
from jupyter_server.base.handlers import APIHandler
from tornado.web import HTTPError, authenticated

from agent_kernel.api import AgentKernel
from agent_kernel.models.budget import Budget


def _agent(handler: APIHandler) -> AgentKernel:
    return handler.settings["agent_kernel"]


def _serialize(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


class _BaseHandler(APIHandler):
    """Common helpers: JSON read + write."""

    # Required by jupyter_server.auth.@authorized — names the resource the
    # configured Authorizer is asked about.
    auth_resource = "agent_kernel"

    def write_json(self, payload: Any, status: int = 200) -> None:
        self.set_status(status)
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps(payload, default=_serialize))

    def read_json(self) -> dict[str, Any]:
        if not self.request.body:
            return {}
        try:
            return json.loads(self.request.body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPError(400, f"invalid JSON: {exc}") from exc


class TasksHandler(_BaseHandler):
    """``POST /api/agent-kernel/tasks`` — create a task."""

    @authenticated
    @authorized
    def post(self) -> None:
        body = self.read_json()
        notebook_path = body.get("notebook_path")
        if not notebook_path:
            raise HTTPError(400, "notebook_path is required")
        kwargs: dict[str, Any] = {
            "notebook_path": notebook_path,
            "kernel_name": body.get("kernel_name", "python3"),
        }
        if "parameters" in body:
            kwargs["parameters"] = body["parameters"]
        if "reserved_budget" in body:
            kwargs["reserved_budget"] = Budget.model_validate(body["reserved_budget"])
        if "tags" in body:
            kwargs["tags"] = body["tags"]
        task = _agent(self).create_task(**kwargs)
        self.write_json({"task": task.model_dump(mode="json")}, status=201)


class TaskHandler(_BaseHandler):
    """``GET /api/agent-kernel/tasks/{id}`` — fetch a task."""

    @authenticated
    @authorized
    def get(self, task_id: str) -> None:
        t = _agent(self).get_task(task_id)
        if t is None:
            raise HTTPError(404, f"task {task_id!r} not found")
        self.write_json({"task": t.model_dump(mode="json")})


class TaskRunHandler(_BaseHandler):
    """``POST /api/agent-kernel/tasks/{id}/run`` — run a task to completion."""

    @authenticated
    @authorized
    async def post(self, task_id: str) -> None:
        ak = _agent(self)
        if ak.get_task(task_id) is None:
            raise HTTPError(404, f"task {task_id!r} not found")
        # Drive the async scheduler directly to avoid AgentKernel.run_task's
        # asyncio.run() (we're already inside Tornado's loop).
        final = await ak.scheduler.run_task(task_id)
        self.write_json({"task": final.model_dump(mode="json")})


class TaskEventsHandler(_BaseHandler):
    """``GET /api/agent-kernel/tasks/{id}/events`` — list provenance events."""

    @authenticated
    @authorized
    def get(self, task_id: str) -> None:
        events = _agent(self).list_events(task_id)
        self.write_json(
            {"events": [e.model_dump(mode="json") for e in events], "count": len(events)}
        )


class HealthHandler(_BaseHandler):
    """``GET /api/agent-kernel/health`` — unauthenticated liveness probe."""

    def get(self) -> None:
        # Intentionally not @authenticated — used by tests + monitoring.
        self.write_json({"ok": True, "service": "agent-kernel"})
