"""DOM-based verification — evaluation-only, used to confirm/score a
run's outcome, never as runtime verification. Runtime verification is
pixels-first (visipilot/action/verification.py); this module exists only
so an eval harness can cross-check the pixels-first verdict against
ground truth, exactly like visipilot/eval/dom_ground_truth.py does for
grounding. Never imported by runtime pipeline code.
"""
from __future__ import annotations

from playwright.sync_api import Page


def dom_text_present(page: Page, selector: str, expected_substring: str) -> bool:
    """True if the element matching `selector` currently contains
    `expected_substring` in its rendered text (case-insensitive).
    """
    element = page.query_selector(selector)
    if element is None:
        return False
    text = element.inner_text() or ""
    return expected_substring.lower() in text.lower()
