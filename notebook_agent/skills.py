"""Filesystem-backed skill discovery (no scoring — selection is a DSPy step).

A skill is any directory containing a ``manifest.json`` (and optionally a
``SKILL.md`` and ``skill.ipynb``). This module is purely concerned with
**finding** skills on disk and turning them into structured records; choosing
*which* skill should handle a request is the job of
:class:`notebook_agent.program.NotebookAgentProgram.skill_chooser`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Default search paths (relative to a project root).
DEFAULT_SKILL_DIRS: tuple[str, ...] = ("skills",)


@dataclass
class Skill:
    """A loaded skill record."""

    skill_id: str
    name: str
    directory: Path
    manifest: dict[str, Any]
    skill_md: str
    notebook_path: Path | None = None

    @property
    def version(self) -> str:
        return str(self.manifest.get("version", "0.0.0"))

    @property
    def description(self) -> str:
        return str(self.manifest.get("description", ""))

    @property
    def tags(self) -> list[str]:
        return [str(t) for t in (self.manifest.get("tags") or [])]

    @property
    def entrypoint(self) -> str | None:
        ep = self.manifest.get("entrypoint")
        return str(ep) if ep else None

    def excerpt(self, max_chars: int = 280) -> str:
        text = self.skill_md.strip()
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

    def catalog_entry(self) -> dict[str, Any]:
        """Compact record handed to the DSPy chooser as part of the catalog."""
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "directory": str(self.directory),
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
            "entrypoint": self.entrypoint,
            "excerpt": self.excerpt(),
        }


class SkillRepository:
    """Filesystem-backed local skill repository.

    A skill is any directory containing a ``manifest.json`` (and optionally a
    ``SKILL.md``). The repository searches its configured roots recursively
    and returns :class:`Skill` records. **There is no lexical scoring** — pass
    :meth:`catalog` to a DSPy chooser to pick a skill.
    """

    def __init__(self, roots: list[Path | str] | None = None) -> None:
        if roots is None:
            roots = list(DEFAULT_SKILL_DIRS)
        self.roots: list[Path] = [Path(r) for r in roots]

    # ---------------- discovery ----------------

    def iter_skill_dirs(self) -> list[Path]:
        out: list[Path] = []
        seen: set[Path] = set()
        for root in self.roots:
            if not root.exists():
                continue
            for manifest in root.rglob("manifest.json"):
                d = manifest.parent
                if d in seen:
                    continue
                seen.add(d)
                out.append(d)
        return sorted(out)

    def load_skill(self, directory: Path) -> Skill | None:
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        skill_md_path = directory / "SKILL.md"
        skill_md = skill_md_path.read_text(encoding="utf-8") if skill_md_path.exists() else ""
        nb = directory / (manifest.get("entrypoint") or "skill.ipynb")
        return Skill(
            skill_id=str(manifest.get("skill_id") or directory.name),
            name=str(manifest.get("name") or directory.name),
            directory=directory,
            manifest=manifest,
            skill_md=skill_md,
            notebook_path=nb if nb.exists() else None,
        )

    def all_skills(self) -> list[Skill]:
        skills: list[Skill] = []
        for d in self.iter_skill_dirs():
            s = self.load_skill(d)
            if s is not None:
                skills.append(s)
        return skills

    def find(self, skill_id: str) -> Skill | None:
        for s in self.all_skills():
            if s.skill_id == skill_id or s.directory.name == skill_id:
                return s
        return None

    # ---------------- catalog for DSPy chooser ----------------

    def catalog(self) -> list[dict[str, Any]]:
        """Compact JSON-serialisable catalog for the DSPy ``ChooseSkill`` step."""
        return [s.catalog_entry() for s in self.all_skills()]

    def catalog_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.catalog(), indent=indent, sort_keys=False)


__all__ = ["DEFAULT_SKILL_DIRS", "Skill", "SkillRepository"]
