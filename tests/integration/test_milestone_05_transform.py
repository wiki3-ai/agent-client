"""Milestone 5 acceptance test: SKILL.md -> notebook transform + execute."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

from notebook_agent import Budget, BudgetTracker, create_root_task
from notebook_agent.notebook_exec import execute_notebook
from notebook_agent.skills import SkillRepository
from notebook_agent.transform import builtin_skills_root, transform_skill_to_notebook


def _write_skill(root: Path, *, skill_id: str, name: str, md: str, input_schema: dict) -> Path:
    d = root / skill_id.split(".")[-1]
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "skill_id": skill_id,
                "name": name,
                "version": "0.1.0",
                "entrypoint": "skill.ipynb",
                "description": f"{name} skill",
                "input_schema": input_schema,
                "tags": [],
            },
            indent=2,
        )
    )
    (d / "SKILL.md").write_text(md)
    return d


def test_transform_simple_echo_md_and_execute(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        skill_id="local.echo",
        name="Local Echo",
        md=(
            "# Echo\n\n"
            "Return the input message.\n\n"
            "## Implementation\n\n"
            "```python\n"
            "result = {'message': message}\n"
            "```\n"
        ),
        input_schema={"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
    )

    skill = SkillRepository([skills_root]).find("local.echo")
    assert skill is not None

    task = create_root_task(tmp_path / "runs", title="Echo task", request="echo")
    nb_path = transform_skill_to_notebook(skill, task.task_notebook)
    assert nb_path.exists()

    # The notebook must have a Papermill `parameters` cell.
    nb = nbformat.read(str(nb_path), as_version=4)
    param_cells = [c for c in nb.cells if "parameters" in (c.metadata.get("tags") or [])]
    assert len(param_cells) == 1
    assert "message" in param_cells[0].source

    # Execute and check output.
    res = execute_notebook(
        nb_path,
        parameters={"message": "ahoy"},
        output_path=task.executed_notebook,
        run_dir=task.directory,
        event_log=task.event_log(),
        budget=BudgetTracker(Budget()),
    )
    assert res.success, res.error
    assert task.result_json.exists()
    payload = json.loads(task.result_json.read_text())
    assert payload == {"message": "ahoy"}


def test_transform_default_body_echoes_inputs(tmp_path: Path) -> None:
    """A SKILL.md with no `## Implementation` block still produces a working notebook."""
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        skill_id="local.bare_echo",
        name="Bare Echo",
        md="# Bare\n\nNo implementation block. The default body echoes inputs.\n",
        input_schema={"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
    )
    skill = SkillRepository([skills_root]).find("local.bare_echo")
    assert skill is not None
    task = create_root_task(tmp_path / "runs", title="Bare echo", request="echo")
    nb_path = transform_skill_to_notebook(skill, task.task_notebook)
    res = execute_notebook(
        nb_path,
        parameters={"message": "hi"},
        output_path=task.executed_notebook,
        run_dir=task.directory,
        event_log=task.event_log(),
        budget=BudgetTracker(Budget()),
    )
    assert res.success, res.error
    assert json.loads(task.result_json.read_text()) == {"message": "hi"}


def test_transform_builtin_echo_skill(tmp_path: Path) -> None:
    repo = SkillRepository([builtin_skills_root()])
    skill = repo.find("core.echo")
    assert skill is not None
    task = create_root_task(tmp_path / "runs", title="Builtin echo", request="echo hello")
    nb_path = transform_skill_to_notebook(skill, task.task_notebook)
    res = execute_notebook(
        nb_path,
        parameters={"message": "hello graph agent"},
        output_path=task.executed_notebook,
        run_dir=task.directory,
        event_log=task.event_log(),
        budget=BudgetTracker(Budget()),
    )
    assert res.success, res.error
    assert json.loads(task.result_json.read_text()) == {"message": "hello graph agent"}
    # Manifest was updated with skill id and outputs.
    m = json.loads(task.manifest_json.read_text())
    assert m["outputs"]["result_json"].endswith("result.json")
    assert m["notebook"]["skill_id"] == "core.echo"
