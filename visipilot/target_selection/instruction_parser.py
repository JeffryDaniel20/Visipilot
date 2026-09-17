"""Instruction parsing: split a natural-language instruction into a
sequence of typed steps.

Deliberately rule-based, not an LLM — Instructions.md's pipeline-over-VLM
default bias and model-selection rule mean a language model is only
justified once evaluation shows rule-based parsing is insufficient for
the instruction shapes this project actually targets, not before.
"""
from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel


class ActionKind(str, Enum):
    FIND = "find"
    CLICK = "click"
    TYPE = "type"


class InstructionStep(BaseModel):
    action: ActionKind
    target_phrase: str | None = None  # for FIND / CLICK
    value: str | None = None  # for TYPE


_CLAUSE_SPLIT_RE = re.compile(r",\s*(?:and\s+)?|\s+and\s+", re.IGNORECASE)
_FIND_RE = re.compile(r"^(?:find|locate)\s+(?:the\s+)?(.+)$", re.IGNORECASE)
_CLICK_RE = re.compile(r"^click(?:\s+on)?\s+(?:the\s+)?(.+)$", re.IGNORECASE)
_TYPE_RE = re.compile(r"^type\s+(.+?)(?:\s+into\s+.+)?$", re.IGNORECASE)


def parse_instruction(instruction: str) -> list[InstructionStep]:
    """Split `instruction` into clauses and classify each as find/click/type.

    A clause that doesn't match a known pattern is silently dropped, not
    raised — an unparseable clause becomes "no step", and downstream
    (target selection returning no candidates for a missing step) already
    has a defined, safe behavior for that, so a separate error path here
    would just be an extra way to fail without adding safety.
    """
    text = instruction.strip().rstrip(".")
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]

    steps: list[InstructionStep] = []
    for clause in clauses:
        m = _TYPE_RE.match(clause)
        if m:
            steps.append(InstructionStep(action=ActionKind.TYPE, value=m.group(1).strip()))
            continue
        m = _FIND_RE.match(clause)
        if m:
            steps.append(InstructionStep(action=ActionKind.FIND, target_phrase=m.group(1).strip()))
            continue
        m = _CLICK_RE.match(clause)
        if m:
            steps.append(InstructionStep(action=ActionKind.CLICK, target_phrase=m.group(1).strip()))
            continue

    return steps
