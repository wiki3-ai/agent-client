"""Milestone 3 acceptance test: local skill search/retrieval."""

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


def test_search_ranks_execute_notebook_first(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills" / "core"
    _make_skill(
        skills_root,
        "core.execute_notebook",
        "Execute Notebook",
        "Execute a parameterized Jupyter notebook using Papermill.",
        "# Execute Notebook\n\nRun papermill on a target notebook with given parameters.",
        ["core", "papermill", "execution"],
    )
    _make_skill(
        skills_root,
        "core.search_local",
        "Search Local",
        "Search local files and skills by lexical match.",
        "# Search Local\n\nLexical local search.",
        ["core", "search"],
    )
    _make_skill(
        skills_root,
        "core.summarize_result",
        "Summarize Result",
        "Summarize a JSON result into a human answer.",
        "# Summarize\n\nGenerate a summary.",
        ["core", "summarize"],
    )

    repo = SkillRepository([tmp_path / "skills"])
    results = repo.search("run papermill notebook")
    assert results, "expected at least one match"
    assert results[0].skill.skill_id == "core.execute_notebook"
    # Returned result includes manifest metadata and SKILL.md excerpt.
    top = results[0].to_dict()
    assert top["skill"]["description"].startswith("Execute")
    assert "papermill" in top["skill"]["excerpt"].lower()
    assert "papermill" in top["matched_terms"]


def test_load_individual_skill(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _make_skill(
        skills_root / "core",
        "core.echo",
        "Echo",
        "Echo input.",
        "# Echo\n\nReturn the input message.",
        ["core", "echo"],
    )
    repo = SkillRepository([skills_root])
    s = repo.find("core.echo")
    assert s is not None
    assert s.name == "Echo"
    assert s.tags == ["core", "echo"]


def test_builtin_echo_skill_discoverable() -> None:
    """The packaged echo skill (§24) should be present and searchable."""
    here = Path(__file__).resolve().parents[2] / "notebook_agent" / "builtin_skills"
    repo = SkillRepository([here])
    results = repo.search("echo message")
    assert results
    assert results[0].skill.skill_id == "core.echo"
