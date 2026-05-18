"""Milestone 1 acceptance test: filesystem task graph."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from notebook_agent import Budget, TaskGraph, create_root_task


def test_root_task_directory_layout(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    task = create_root_task(
        runs_root,
        title="Build MCP notebook agent",
        request="Construct a graph agent.",
        budget={"max_notebook_executions": 3},
    )
    # Directory follows YYYY/MM/DD/HHMMSS-slug.
    rel = task.directory.relative_to(runs_root)
    parts = rel.parts
    assert len(parts) == 4, f"expected YYYY/MM/DD/HHMMSS-slug, got {parts}"
    assert parts[0].isdigit() and len(parts[0]) == 4
    assert parts[1].isdigit() and len(parts[1]) == 2
    assert parts[2].isdigit() and len(parts[2]) == 2
    assert parts[3].startswith(parts[3][:6])
    assert "build-mcp-notebook-agent" in parts[3]

    # Required files / directories (Section 8).
    for required in (
        task.task_json,
        task.manifest_json,
        task.readme,
        task.request_md,
        task.parameters_json,
        task.events_log,
    ):
        assert required.exists(), f"missing {required}"
    for required_dir in (task.inputs_dir, task.outputs_dir, task.logs_dir, task.artifacts_dir, task.children_dir):
        assert required_dir.is_dir(), f"missing dir {required_dir}"

    # task.json and manifest.json are valid JSON with required keys.
    tj = json.loads(task.task_json.read_text())
    assert tj["task_id"] == task.task_id
    assert tj["parent_task_id"] is None
    assert tj["title"] == "Build MCP notebook agent"
    assert tj["status"] == "pending"
    assert tj["budget"]["max_notebook_executions"] == 3
    assert tj["created_at"]

    mj = json.loads(task.manifest_json.read_text())
    assert mj["task_id"] == task.task_id
    assert mj["status"] == "pending"
    assert mj["stage_decision"]["chosen"] is None
    assert mj["budget_initial"]["max_notebook_executions"] == 3

    # events.jsonl contains a task_created event.
    events = [json.loads(l) for l in task.events_log.read_text().splitlines() if l.strip()]
    assert any(e["event"] == "task_created" and e["task_id"] == task.task_id for e in events)
    assert any(e["event"] == "budget_allocated" for e in events)


def test_child_task_creation_and_parent_link(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    parent = create_root_task(runs_root, title="Parent", request="Parent request.")
    child = parent.create_child("First Child Step", "Do the first thing.")

    # Child directory name starts with numeric prefix.
    assert child.directory.parent == parent.children_dir
    assert child.directory.name.startswith("001-")
    assert "first-child-step" in child.directory.name

    # Child references parent task ID.
    cj = json.loads(child.task_json.read_text())
    assert cj["parent_task_id"] == parent.task_id

    # Parent manifest lists the child.
    pm = json.loads(parent.manifest_json.read_text())
    assert any(c["task_id"] == child.task_id for c in pm["children"])

    # Child has its own task_created event.
    child_events = [json.loads(l) for l in child.events_log.read_text().splitlines() if l.strip()]
    assert any(e["event"] == "task_created" for e in child_events)

    # Parent's events list the child creation.
    parent_events = [json.loads(l) for l in parent.events_log.read_text().splitlines() if l.strip()]
    assert any(e["event"] == "child_task_created" and e["child_task_id"] == child.task_id for e in parent_events)


def test_multiple_children_get_sequential_prefixes(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    parent = create_root_task(runs_root, title="Parent2", request="r")
    c1 = parent.create_child("Step One", "x")
    c2 = parent.create_child("Step Two", "y")
    c3 = parent.create_child("Step Three", "z")
    assert c1.directory.name.startswith("001-")
    assert c2.directory.name.startswith("002-")
    assert c3.directory.name.startswith("003-")


def test_reload_graph_from_disk(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    parent = create_root_task(runs_root, title="Reload Me", request="r", budget=Budget(max_subtasks=5))
    parent.create_child("a", "ra")
    grand_parent_child = parent.create_child("b", "rb")
    grand_parent_child.create_child("grand", "rg")

    graph = TaskGraph.load(parent.directory)
    walked = graph.walk()
    assert len(walked) == 4
    assert graph.root.task_id == parent.task_id
    # Order of root.children is by directory name (001-, 002-).
    assert [c.directory.name for c in graph.root.children] == sorted(
        c.directory.name for c in graph.root.children
    )
    # Grandchild is reachable through traversal.
    assert any(t.title == "grand" for t in walked)


def test_invalid_status_rejected(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    task = create_root_task(runs_root, title="t", request="r")
    with pytest.raises(ValueError):
        task.update_status("bogus")
    task.update_status("running")
    assert json.loads(task.task_json.read_text())["status"] == "running"
