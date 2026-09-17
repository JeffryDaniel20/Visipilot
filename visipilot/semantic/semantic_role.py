"""Lightweight semantic_role inference: a keyword + type heuristic, not a
trained classifier. Deliberately simple — Instructions.md's "prefer the
simplest technically sound implementation" and "no speculative
abstractions" rules mean a real classifier is only justified once
evaluation shows this heuristic is insufficient, not before.
"""
from __future__ import annotations

from visipilot.types import ElementType, UIElement

_SEARCH_WORDS = {"search", "find", "query"}
_SUBMIT_WORDS = {"submit", "subscribe", "send", "go", "ok", "confirm"}
_INPUT_WORDS = {"email", "password", "username", "name"}


def infer_semantic_role(element: UIElement) -> str | None:
    text = (element.text or "").strip().lower()
    words = set(text.replace(":", "").replace("...", "").split())

    if not words:
        return None

    if words & _SEARCH_WORDS:
        if element.type == ElementType.BUTTON:
            return "search-button"
        if element.type == ElementType.TEXT_INPUT:
            return "search-input"
        return "search-related"

    if words & _SUBMIT_WORDS and element.interactable:
        return "submit-button"

    if words & _INPUT_WORDS:
        return "labeled-input"

    return None
