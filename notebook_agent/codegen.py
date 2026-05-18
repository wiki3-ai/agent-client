"""LLM-driven code generation for the Generate stage.

When no local skill matches the user's request, the agent asks the LLM to
write a small Python snippet that solves the task and binds its answer to a
local variable named ``result``. The snippet is then wrapped in a Papermill
notebook (via :func:`notebook_agent.codegen.build_generated_notebook`) and
executed.

Design notes
------------
* The LLM contract is intentionally narrow: produce a fenced ``python`` block
  whose body sets ``result`` to a JSON-serialisable value (dict, list, scalar,
  or string). No imports of third-party packages — only the standard library.
* If the model wraps its answer in prose, we strip code fences and accept the
  first/last fenced block. If parsing fails, we fall through to a deterministic
  no-LLM fallback so the agent still produces a manifest and a "failed" answer
  rather than crashing.
* ``provider="fake"`` is fully supported by passing a canned snippet through
  ``LiteLLMClient.fake_response``; this is how the integration tests exercise
  the Generate path without LM Studio.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ._notebook_io import (
    DEFAULT_KERNEL_NAME,
    make_notebook,
    parameters_cell,
    write_notebook,
    write_result_cell,
)
from .litellm_client import LiteLLMClient, LLMUnavailableError

SYSTEM_PROMPT = (
    "You write small Python snippets that solve a user's task. "
    "Output exactly one fenced ``python`` block and nothing else: no prose, no commentary. "
    "The snippet must bind its answer to a variable named `result`. "
    "`result` must be JSON-serialisable (dict, list, int, float, str, or bool). "
    "Use only the Python standard library. Do not call input(); do not access the network; "
    "do not write files. The snippet runs inside a Jupyter notebook cell."
)

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(?P<code>.*?)\n```", re.DOTALL | re.IGNORECASE)


@dataclass
class GeneratedCode:
    """A single LLM-generated Python snippet for the Generate stage."""

    source: str
    raw_response: str
    prompt: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "raw_response": self.raw_response, "prompt": self.prompt}


def _strip_fences(text: str) -> str:
    """Pull the first ```python ... ``` block out of *text*, or return text as-is."""
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group("code").strip()
    # No fence — accept the whole response as code if it parses; otherwise
    # strip a leading "python" marker the model may have added.
    s = text.strip()
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _validate_snippet(source: str) -> None:
    """Cheap static checks before we hand the code to Papermill."""
    if not source or not source.strip():
        raise ValueError("LLM returned an empty snippet")
    compile(source, "<generated>", "exec")  # raises SyntaxError on bad code
    if "result" not in source:
        raise ValueError("generated snippet does not assign `result`")


def generate_code_for_request(
    request: str,
    *,
    llm: LiteLLMClient,
    max_tokens: int = 1024,
    extra_context: str | None = None,
) -> GeneratedCode:
    """Ask the LLM to produce a Python snippet that solves *request*.

    Raises :class:`~notebook_agent.litellm_client.LLMUnavailableError` if the
    LLM call fails or :class:`ValueError` if the response cannot be parsed
    into a valid snippet.
    """
    prompt_parts = [
        "Task: " + request.strip(),
        "",
        "Write a Python snippet that solves this task. Bind your answer to "
        "`result`. Standard library only.",
    ]
    if extra_context:
        prompt_parts.extend(["", "Context:", extra_context])
    prompt = "\n".join(prompt_parts)
    resp = llm.complete(prompt, system=SYSTEM_PROMPT, max_tokens=max_tokens)
    source = _strip_fences(resp.text)
    _validate_snippet(source)
    return GeneratedCode(source=source, raw_response=resp.text, prompt=prompt)


def build_generated_notebook(
    request: str,
    code: GeneratedCode,
    output_path: Path | str,
    *,
    plan: list[str] | None = None,
) -> Path:
    """Wrap *code* in a self-contained parameterized notebook.

    The notebook has:
    1. A title/markdown cell describing the task and the plan.
    2. The Papermill ``parameters`` cell (so it's executable headless).
    3. A setup cell that ensures ``output_dir`` exists.
    4. The generated code cell (tagged ``generated``).
    5. The standard write-result cell that dumps ``result`` to
       ``outputs/result.json``.
    6. A tiny smoke cell that asserts ``result.json`` exists.

    The generated code source is also stored on notebook metadata under
    ``notebook_agent.generated.source`` so downstream consumers can read it
    without re-parsing cells.
    """
    output_path = Path(output_path)
    plan_md = ""
    if plan:
        plan_md = "\n\n## Plan\n\n" + "\n".join(f"- [ ] {item}" for item in plan)

    setup_src = (
        "import json\n"
        "from pathlib import Path\n"
        "Path(output_dir).mkdir(parents=True, exist_ok=True)\n"
    )
    body_src = (
        f"# --- generated code for: {request!r} ---\n"
        f"{code.source.rstrip()}\n"
        "if not isinstance(result, dict):\n"
        "    result = {'value': result}\n"
    )
    smoke_src = (
        "from pathlib import Path\n"
        "assert (Path(output_dir) / 'result.json').exists(), 'result.json not written'\n"
    )

    cells = [
        ("markdown", f"# Generated task\n\n**Request:** {request}{plan_md}", []),
        parameters_cell([]),
        ("code", setup_src, ["setup"]),
        ("code", body_src, ["generated"]),
        write_result_cell(),
        ("code", smoke_src, ["smoke"]),
    ]
    nb = make_notebook(cells)
    # Store provenance on notebook metadata.
    nb.metadata.setdefault("notebook_agent", {})
    nb.metadata["notebook_agent"].update(
        {
            "stage": "generate",
            "request": request,
            "generated": code.to_dict(),
            "plan": list(plan or []),
            "kernel": DEFAULT_KERNEL_NAME,
        }
    )
    return write_notebook(nb, output_path)


__all__ = [
    "GeneratedCode",
    "LLMUnavailableError",
    "build_generated_notebook",
    "generate_code_for_request",
]
