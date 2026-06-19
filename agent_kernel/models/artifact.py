"""Executable Cell Artifact: the durable IR linking notebook execution to KG.

This is the smallest portable unit of "autoformalized into executable code"
work. It records the source code, the notebook context, the execution
contract, and provenance anchors.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FormalizationLevel(StrEnum):
    free_text = "free_text"
    executable = "executable"
    checkable = "checkable"
    certified = "certified"


class ExecutableCellArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    task_id: str
    notebook_path: str
    cell_id: str
    kernel_name: str
    language: str
    source: str
    normalized_source: str
    formalization_level: FormalizationLevel = FormalizationLevel.executable
    input_bindings: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)
    side_effects: list[str] = Field(default_factory=list)
    semantic_refs: list[str] = Field(default_factory=list)
    provenance_event_ids: list[str] = Field(default_factory=list)
    content_hash: str
