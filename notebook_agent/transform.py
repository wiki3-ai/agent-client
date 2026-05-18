"""Convert a SKILL.md (+ manifest) into a parameterized notebook (Section 14.7).

The transformer is intentionally template-based. The key idea is:

* A skill ships with a ``manifest.json`` (declaring inputs/outputs) and a
  ``SKILL.md`` document.
* The transformer reads the skill, picks up the input parameter names from
  ``manifest.input_schema.properties``, picks up an optional Python
  implementation snippet from a fenced ``python`` block under an
  ``## Implementation`` heading in ``SKILL.md``, and emits a notebook that:
    1. has a Papermill ``parameters`` cell exposing those inputs;
    2. runs the implementation snippet (or a default echo-style body);
    3. writes ``outputs/result.json``;
    4. updates ``manifest.json`` (best-effort) with a ``transform_completed``
       flag.

This is deliberately simple; it covers the built-in echo skill and any other
skill whose body fits inside a single Python fenced block.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ._notebook_io import make_notebook, parameters_cell, write_notebook, write_result_cell
from .skills import Skill

_IMPL_RE = re.compile(
    r"^##+\s*Implementation\s*$.*?```(?:python|py)\s*\n(?P<code>.*?)\n```",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)


def _extract_implementation(skill_md: str) -> str | None:
    """Pull a Python fenced block out of an `## Implementation` section."""
    m = _IMPL_RE.search(skill_md or "")
    if not m:
        return None
    return m.group("code").strip()


def _input_parameters(skill: Skill) -> dict[str, Any]:
    schema = skill.manifest.get("input_schema") or {}
    if isinstance(schema, str):
        # External schema reference: we cannot follow it here; treat as empty.
        return {}
    props = (schema or {}).get("properties") or {}
    out: dict[str, Any] = {}
    for name, spec in props.items():
        if not isinstance(spec, dict):
            out[name] = None
            continue
        default = spec.get("default")
        if default is None:
            t = spec.get("type")
            default = {
                "string": "",
                "integer": 0,
                "number": 0.0,
                "boolean": False,
                "array": [],
                "object": {},
            }.get(t, None)
        out[name] = default
    return out


def _default_body(input_params: dict[str, Any]) -> str:
    """Default body when SKILL.md has no implementation block.

    Echoes the input parameters back into ``result`` (this is what the built-in
    echo skill relies on).
    """
    if not input_params:
        return "result = {}"
    keys = ", ".join(f"{k!r}: {k}" for k in input_params)
    return f"result = {{{keys}}}"


def transform_skill_to_notebook(
    skill: Skill,
    output_path: Path | str,
    *,
    extra_parameter_lines: list[str] | None = None,
) -> Path:
    """Transform ``skill`` into a notebook at ``output_path``.

    Returns the path to the written notebook.
    """
    output_path = Path(output_path)
    input_params = _input_parameters(skill)
    impl = _extract_implementation(skill.skill_md)

    # Extra parameter lines: one per declared input.
    param_lines: list[str] = list(extra_parameter_lines or [])
    for name, default in input_params.items():
        param_lines.append(f"{name} = {json.dumps(default)}")

    body = impl if impl is not None else _default_body(input_params)
    # If body doesn't bind `result`, treat the last expression's value as result.
    # We keep it simple: require user impl to bind `result`. For the default we
    # already do.

    setup_src = (
        "import json\n"
        "from pathlib import Path\n"
        "# Ensure output_dir exists before any cell that writes to it.\n"
        "Path(output_dir).mkdir(parents=True, exist_ok=True)\n"
    )

    validate_src = (
        "_inputs = {" + ", ".join(f"{k!r}: {k}" for k in input_params) + "}\n"
        "for _k, _v in _inputs.items():\n"
        "    if _v is None:\n"
        "        raise ValueError(f'Missing required parameter: {_k}')\n"
    ) if input_params else "# no declared inputs\n"

    body_src = (
        f"# --- skill body ({skill.skill_id}) ---\n"
        f"{body}\n"
        "if not isinstance(result, dict):\n"
        "    result = {'value': result}\n"
    )

    manifest_update_src = (
        "import json\n"
        "from pathlib import Path\n"
        "_run_dir = Path(run_dir)\n"
        "_manifest_path = _run_dir / 'manifest.json'\n"
        "if _manifest_path.exists():\n"
        "    _m = json.loads(_manifest_path.read_text())\n"
        "    _m.setdefault('outputs', {})\n"
        "    _m['outputs']['result_json'] = str(Path(output_dir, 'result.json'))\n"
        "    _m.setdefault('notebook', {})\n"
        f"    _m['notebook']['skill_id'] = {json.dumps(skill.skill_id)}\n"
        "    _manifest_path.write_text(json.dumps(_m, indent=2))\n"
    )

    smoke_src = (
        "assert isinstance(result, dict), 'result must be a dict'\n"
        "assert (Path(output_dir) / 'result.json').exists(), 'result.json not written'\n"
    )

    cells = [
        ("markdown", f"# {skill.name}\n\n{skill.description or ''}\n\nGenerated from `{skill.skill_id}`.", []),
        parameters_cell(param_lines),
        ("code", setup_src, ["setup"]),
        ("code", validate_src, ["validate"]),
        ("code", body_src, ["execute"]),
        write_result_cell(),
        ("code", manifest_update_src, ["manifest_update"]),
        ("code", smoke_src, ["smoke"]),
    ]
    nb = make_notebook(cells)
    return write_notebook(nb, output_path)


def builtin_skills_root() -> Path:
    """Path to the built-in skill repository shipped with the package."""
    return Path(__file__).resolve().parent / "builtin_skills"
