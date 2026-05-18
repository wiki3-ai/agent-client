"""Skill discovery test.

Selection (which skill handles a given request) is a DSPy step exercised in
``test_dspy_program.py``; this test covers only on-disk discovery + loading.
"""

from __future__ import annotations

import json
from pathlib import Path

from notebook_agent.skills import SkillRepository


def _make_skill(root: Path, skill_id: str, name: str, description: str, md: str, tags: list[str]) -> Path:
    d = root / skill_id.split(".")[-1]
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "skill_id": skill_id,
                "name": name,
                "version": "0.1.0",
                "entrypoint": "skill.ipynb",
                "description": description,
                "tags": tags,
            },
            indent=2,
        )
    )
    (d / "SKILL.md").write_text(md)
    return d


def test_repository_discovers_and_loads_skills(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills" / "core"
    _make_skill(
        skills_root, "core.execute_notebook", "Execute Notebook",
        "Execute a parameterized Jupyter notebook using Papermill.",
        "# Execute Notebook\n\nRun papermill on a target notebook.",
        ["core", "papermill"],
    )
    _make_skill(
        skills_root, "core.echo", "Echo",
        "Echo input.", "# Echo\n\nReturn the input message.", ["core", "echo"],
    )
    repo = SkillRepository([tmp_path / "skills"])
    ids = {s.skill_id for s in repo.all_skills()}
    assert ids == {"core.execute_notebook", "core.echo"}
    # Catalog is the structured form fed to the DSPy chooser.
    catalog = repo.catalog()
    assert all({"skill_id", "name", "description", "tags"} <= set(e) for e in catalog)


def test_load_individual_skill(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _make_skill(
        skills_root / "core", "core.echo", "Echo",
        "Echo input.", "# Echo\n\nReturn the input message.", ["core", "echo"],
    )
    repo = SkillRepository([skills_root])
    s = repo.find("core.echo")
    assert s is not None
    assert s.name == "Echo"
    assert s.tags == ["core", "echo"]


def test_builtin_echo_skill_discoverable() -> None:
    here = Path(__file__).resolve().parents[2] / "notebook_agent" / "builtin_skills"
    repo = SkillRepository([here])
    ids = {s.skill_id for s in repo.all_skills()}
    assert "core.echo" in ids
