"""Live progress display for the ``%task`` / ``%%task`` magics.

Subscribes to the :mod:`notebook_agent.events` bus, converts incoming
events into human-readable status lines, and updates a single IPython
display handle in place so the user sees what the agent is doing while
LM calls (which can take many seconds each) are in flight.

Without this, a long ``run_task`` invocation looks like a dead kernel.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .events import subscribe, unsubscribe


def _humanize(event: dict[str, Any]) -> str | None:
    """Map a single event dict to a status line, or ``None`` to skip."""
    name = event.get("event", "")
    if name == "task_created":
        return f"Task created — `{event.get('task_id', '')}`"
    if name == "task_continuation":
        return f"Continuing from `{event.get('parent_task_id', '')}`"
    if name == "task_started":
        return "Task started."
    if name == "plan_created":
        plan = event.get("plan") or []
        if not plan:
            return "Plan: (empty)"
        lines = [f"  {i}. {p}" for i, p in enumerate(plan, 1)]
        return "Plan:\n" + "\n".join(lines)
    if name == "retrieval_started":
        n = event.get("catalog_size", "?")
        return f"Searching skills ({n} candidates)…"
    if name == "retrieval_finished":
        chosen = event.get("chosen")
        return f"Chosen skill: `{chosen}`" if chosen else "No matching skill — will generate."
    if name == "parameters_inferred":
        inferred = event.get("inferred") or {}
        if inferred:
            return f"Inferred parameters: `{', '.join(inferred.keys())}`"
        return "No parameters inferred."
    if name == "generation_started":
        return "Generating notebook code…"
    if name == "generation_finished":
        return f"Generated code ({event.get('source_chars', 0)} chars)."
    if name == "notebook_execution_started":
        return f"Executing notebook `{event.get('path', '')}`…"
    if name == "notebook_execution_finished":
        ok = event.get("success")
        return "Notebook executed successfully." if ok else "Notebook execution failed."
    if name == "lm_call_started":
        return f"Calling LM ({event.get('step', '?')})…"
    if name == "lm_call_finished":
        step = event.get("step", "?")
        elapsed = event.get("elapsed_s", "?")
        return f"LM ({step}) returned in {elapsed}s."
    if name == "lm_call_failed":
        step = event.get("step", "?")
        elapsed = event.get("elapsed_s", "?")
        et = event.get("error_type", "Error")
        return f"LM ({step}) FAILED after {elapsed}s — {et}: {event.get('error', '')}"
    if name == "repair_started":
        return f"Repair attempt: {event.get('strategy', '?')}…"
    if name == "repair_finished":
        return f"Repair {'succeeded' if event.get('success') else 'failed'}."
    if name == "manifest_updated":
        return f"Manifest updated (status={event.get('status', '?')})."
    if name == "task_finished":
        return f"Task finished — status `{event.get('status', '?')}`."
    return None


class ProgressRenderer:
    """Streams agent events into a single updatable IPython display.

    Usage::

        with ProgressRenderer():
            result = run_task(...)

    Outside a notebook, falls back to printing one line per event.
    """

    def __init__(self, *, prefix: str = "🛠️  notebook_agent") -> None:
        self._prefix = prefix
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self._handle = None
        self._fallback_print = False
        self._installed = False
        self._last_step: str | None = None
        self._last_step_t0: float | None = None
        self._stop_pulse = threading.Event()
        self._pulse_thread: threading.Thread | None = None

    # ----- context manager -----

    def __enter__(self) -> "ProgressRenderer":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.uninstall()

    # ----- lifecycle -----

    def install(self) -> None:
        if self._installed:
            return
        self._installed = True
        self._handle = _make_display_handle(self._render())
        if self._handle is None:
            self._fallback_print = True
        subscribe(self._on_event)
        # Background pulse so the user sees a running timer while an LM call
        # is in flight.
        self._pulse_thread = threading.Thread(target=self._pulse_loop, daemon=True)
        self._pulse_thread.start()

    def uninstall(self) -> None:
        if not self._installed:
            return
        unsubscribe(self._on_event)
        self._stop_pulse.set()
        self._installed = False
        # Final render to clear the in-flight marker.
        self._last_step = None
        self._refresh()

    # ----- event handling -----

    def _on_event(self, event: dict[str, Any]) -> None:
        text = _humanize(event)
        if text is None:
            return
        name = event.get("event", "")
        with self._lock:
            elapsed = round(time.monotonic() - self._t0, 1)
            self._lines.append(f"[{elapsed:>5}s] {text}")
            if name == "lm_call_started":
                self._last_step = event.get("step", "?")
                self._last_step_t0 = time.monotonic()
            elif name in {"lm_call_finished", "lm_call_failed"}:
                self._last_step = None
                self._last_step_t0 = None
        self._refresh()

    def _pulse_loop(self) -> None:
        # Update once a second while an LM call is in flight so the user sees
        # the timer climb instead of a frozen "Calling LM …" line.
        while not self._stop_pulse.wait(1.0):
            with self._lock:
                in_flight = self._last_step is not None
            if in_flight:
                self._refresh()

    # ----- rendering -----

    def _render(self) -> Any:
        try:
            from IPython.display import Markdown  # type: ignore[import-not-found]
        except Exception:
            return self._render_text()
        return Markdown(self._render_text())

    def _render_text(self) -> str:
        with self._lock:
            lines = list(self._lines)
            in_flight = self._last_step
            t0 = self._last_step_t0
        body = "\n".join(lines) if lines else "(starting…)"
        header = f"**{self._prefix}**"
        if in_flight is not None and t0 is not None:
            waited = round(time.monotonic() - t0, 1)
            header += f"  — _waiting on LM ({in_flight}) — {waited}s_"
        return f"{header}\n\n```text\n{body}\n```"

    def _refresh(self) -> None:
        if self._fallback_print:
            with self._lock:
                if self._lines:
                    print(self._lines[-1], flush=True)
            return
        if self._handle is None:
            return
        try:
            self._handle.update(self._render())
        except Exception:
            # Display handle may be invalid (kernel restart, etc.).
            self._fallback_print = True


def _make_display_handle(initial: Any) -> Any:
    try:
        from IPython.display import display  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        return display(initial, display_id=True)
    except Exception:
        return None


__all__ = ["ProgressRenderer"]
