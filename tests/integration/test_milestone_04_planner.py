"""Milestone 4 acceptance test: TODO decomposition and child composition."""

from __future__ import annotations

import json
from pathlib import Path

from notebook_agent import create_root_task
from notebook_agent.planner import decompose_request


def test_decompose_request_yields_sequential_todos() -> None:
    todos = decompose_request("search for a skill and execute a notebook")
    assert len(todos) >= 2
    titles = [t.title.lower() for t in todos]
    assert any("search" in t for t in titles)
    assert any("execute" in t for t in titles)


def test_create_children_from_todo_list(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    parent = create_root_task(
        runs_root,
        title="Search and execute",
        request="search for a skill and then execute a notebook",
    )
    todos = decompose_request(parent.request)
    children = [parent.create_child(t.title, t.description) for t in todos]
    assert len(children) >= 2

    # Numeric prefixes in order.
    names = [c.directory.name for c in children]
    for i, n in enumerate(names, start=1):
        assert n.startswith(f"{i:03d}-"), names

    # Parent manifest lists children in order.
    pm = json.loads(parent.manifest_json.read_text())
    listed = [c["task_id"] for c in pm["children"]]
    assert listed == [c.task_id for c in children]


def test_decompose_falls_back_to_single_todo() -> None:
    todos = decompose_request("Echo hello world")
    assert len(todos) == 1
    assert todos[0].description == "Echo hello world"
