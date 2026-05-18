"""Local skill discovery and retrieval (Section 14.5 of the spec)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Default search paths (relative to a project root).
DEFAULT_SKILL_DIRS = ("skills",)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

# Common stopwords that should not contribute to skill-match scores. Without
# this filter generic prompts trivially match any skill whose description
# contains words like "a", "the", "and", "notebook" — which is what made the
# agent silently route every prompt to the echo skill.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "else",
        "of", "in", "on", "at", "to", "for", "from", "by", "with",
        "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "doing",
        "have", "has", "had",
        "this", "that", "these", "those",
        "i", "you", "we", "they", "it", "he", "she",
        "my", "your", "our", "their",
        "as", "so", "not", "no", "yes",
        "can", "will", "would", "should", "may", "might",
        "use", "using", "used",
        "please", "task", "request",
        # Notebook/agent infrastructure words that should not move the score:
        # any task is "in a notebook", so matching on "notebook" or "agent"
        # told us nothing about skill relevance and produced false positives.
        "notebook", "agent", "code", "python",
    }
)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOPWORDS]


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


@dataclass
class SkillSearchResult:
    skill: Skill
    score: float
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "matched_terms": self.matched_terms,
            "skill": self.skill.to_dict(),
        }


class SkillRepository:
    """Filesystem-backed local skill repository.

    A skill is any directory containing a ``manifest.json`` (and optionally a
    ``SKILL.md``). The repository searches its configured roots recursively.
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

    # ---------------- search ----------------

    def search(self, query: str, *, top_k: int | None = None) -> list[SkillSearchResult]:
        q_tokens = set(_tokens(query))
        results: list[SkillSearchResult] = []
        for skill in self.all_skills():
            score, matched = _score_skill(skill, q_tokens)
            if score > 0:
                results.append(SkillSearchResult(skill=skill, score=score, matched_terms=matched))
        results.sort(key=lambda r: (-r.score, r.skill.skill_id))
        if top_k is not None:
            results = results[:top_k]
        return results

    def find(self, skill_id: str) -> Skill | None:
        for s in self.all_skills():
            if s.skill_id == skill_id or s.directory.name == skill_id:
                return s
        return None


def _score_skill(skill: Skill, q_tokens: set[str]) -> tuple[float, list[str]]:
    """Simple lexical scorer with field weights and exact-name boosts.

    A skill only scores when at least one query token hits its name, id, or
    tags — pure description/markdown matches alone are not enough. This
    prevents a skill (e.g. ``core.echo``) from claiming a prompt simply because
    its description happens to mention generic words.
    """
    if not q_tokens:
        return 0.0, []
    name_tokens = set(_tokens(skill.name) + _tokens(skill.directory.name))
    id_tokens = set(_tokens(skill.skill_id))
    desc_tokens = set(_tokens(skill.description))
    tag_tokens = set(_tokens(" ".join(skill.tags)))
    md_tokens = set(_tokens(skill.skill_md))

    score = 0.0
    matched: set[str] = set()
    strong_hit = False
    for tok in q_tokens:
        if tok in name_tokens:
            score += 3.0
            matched.add(tok)
            strong_hit = True
        if tok in id_tokens:
            score += 2.5
            matched.add(tok)
            strong_hit = True
        if tok in tag_tokens:
            score += 2.0
            matched.add(tok)
            strong_hit = True
        if tok in desc_tokens:
            score += 1.5
            matched.add(tok)
        if tok in md_tokens:
            score += 1.0
            matched.add(tok)
    if not strong_hit:
        return 0.0, []
    # Light boost when many query terms match.
    score *= 1.0 + 0.1 * len(matched)
    return score, sorted(matched)
