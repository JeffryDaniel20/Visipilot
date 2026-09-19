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
    # 1-based position among candidates tied for the same phrase (e.g.
    # "the second result"), or -1 for "last" (Python-style negative
    # index — see target_selection.matcher._resolve_ordinal). None means
    # no ordinal reference was made; resolution is unchanged from before
    # ordinal support existed (Instructions.md #7's tie -> refuse path).
    ordinal: int | None = None


_CLAUSE_SPLIT_RE = re.compile(r",\s*(?:and\s+)?|\s+and\s+", re.IGNORECASE)
_FIND_RE = re.compile(r"^(?:find|locate)\s+(?:the\s+)?(.+)$", re.IGNORECASE)
# "open" is accepted as a click synonym -- the roadmap's own canonical
# example ("open the second result") uses it, and in web-UI instructions
# it means exactly "click this to reveal/navigate to it".
_CLICK_RE = re.compile(r"^(?:click(?:\s+on)?|open)\s+(?:the\s+)?(.+)$", re.IGNORECASE)
_TYPE_RE = re.compile(r"^type\s+(.+?)(?:\s+into\s+.+)?$", re.IGNORECASE)

_ORDINAL_WORDS: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
    "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10,
    "last": -1,
}
# Longest words first so e.g. a hypothetical "tenth" isn't cut short by a
# shorter alternative earlier in the alternation.
_ORDINAL_RE = re.compile(
    r"^(" + "|".join(sorted(_ORDINAL_WORDS, key=len, reverse=True)) + r")\s+(.+)$",
    re.IGNORECASE,
)


def _extract_ordinal(phrase: str) -> tuple[str, int | None]:
    """Split a leading ordinal word off `phrase`, returning
    `(base_phrase, ordinal)`. `base_phrase` is what actually gets matched
    against page elements (e.g. "second search button" -> "search
    button", ordinal=2) — the ordinal itself never becomes part of the
    text-matching phrase, since no real element is literally labeled
    "second".

    An ordinal word with nothing after it (just "the first.") isn't
    stripped: there'd be no base phrase left to match against, so
    treating the whole thing as a literal (unmatchable, safely-no-
    candidates) phrase is more honest than guessing what noun was meant.
    """
    m = _ORDINAL_RE.match(phrase.strip())
    if not m:
        return phrase, None
    rest = m.group(2).strip()
    if not rest:
        return phrase, None
    return rest, _ORDINAL_WORDS[m.group(1).lower()]


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
            phrase, ordinal = _extract_ordinal(m.group(1).strip())
            steps.append(InstructionStep(action=ActionKind.FIND, target_phrase=phrase, ordinal=ordinal))
            continue
        m = _CLICK_RE.match(clause)
        if m:
            phrase, ordinal = _extract_ordinal(m.group(1).strip())
            steps.append(InstructionStep(action=ActionKind.CLICK, target_phrase=phrase, ordinal=ordinal))
            continue

    return steps
