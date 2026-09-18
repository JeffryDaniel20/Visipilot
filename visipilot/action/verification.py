"""Verification: confirms an action had its expected effect via
pixels-first visual/OCR diffing on a fresh screenshot — never via the
DOM at runtime (Instructions.md #2/#5). DOM-based confirmation is
eval-only and lives in visipilot/eval/; nothing in this module imports
it.
"""
from __future__ import annotations

from pathlib import Path

from visipilot.perception.ocr import OCREngine
from visipilot.types import UIElement, VerificationResult

# Two OCR regions are treated as the same visual line if their vertical
# centers are within this many pixels — see _reading_order_text().
_LINE_GROUPING_TOLERANCE_PX = 10.0


def _grouped_lines(elements: list[UIElement]) -> list[str]:
    """Groups OCR elements into visual lines by overlapping vertical
    center, sorts each line left-to-right, and returns one joined string
    per line — deliberately NOT one single joined string for the whole
    page, so text from unrelated, distant parts of a page can never
    concatenate into a false match (a real risk an early version of this
    function had, caught by test_verify_text_present_does_not_merge_
    separate_lines before it shipped).

    Exists because EasyOCR does not always merge a single visual line
    into one detection — real, observed behavior on this project's own
    Bootstrap-style test page: "Results for: Python" came back as two
    separate regions, "Results for:" and "Python", with matching
    vertical centers (~349px) but different y-tops (341 vs 335, from
    differing detected text heights). A naive sort by `bbox.y` alone can
    reverse such pairs; grouping by vertical center first does not. See
    implementation-plan.md Phase B for the investigation this came from.
    """
    items = [(el.bbox.y + el.bbox.height / 2, el.bbox.x, el.text) for el in elements if el.text]
    items.sort(key=lambda t: (t[0], t[1]))

    lines: list[list[tuple[float, float, str]]] = []
    for center_y, x, text in items:
        if lines and abs(center_y - lines[-1][-1][0]) < _LINE_GROUPING_TOLERANCE_PX:
            lines[-1].append((center_y, x, text))
        else:
            lines.append([(center_y, x, text)])

    return [" ".join(t[2] for t in sorted(line, key=lambda t: t[1])) for line in lines]


def verify_text_present(ocr: OCREngine, screenshot_image_path: str | Path, expected_substring: str) -> VerificationResult:
    """Re-runs OCR on a screenshot taken after an action and checks
    whether `expected_substring` appears anywhere in the read text
    (case-insensitive). This is the general-purpose pixels-first check
    for "did the page visibly show the result I expected" — e.g. a
    search-results message appearing after a search action.

    Checks each OCR region individually first (cheap, common case), then
    falls back to checking each grouped visual *line* (never the whole
    page joined together) — so a phrase EasyOCR happens to split across
    multiple boxes on the same line is still found, without risking a
    false match built from unrelated text elsewhere on the page.
    """
    elements = ocr.read(screenshot_image_path)
    needle = expected_substring.lower()

    for el in elements:
        if el.text and needle in el.text.lower():
            return VerificationResult(
                passed=True,
                method="ocr_text_present",
                detail=f"found {expected_substring!r} in a single OCR region: {el.text!r}",
            )

    for line in _grouped_lines(elements):
        if needle in line.lower():
            return VerificationResult(
                passed=True,
                method="ocr_text_present_joined",
                detail=f"found {expected_substring!r} across multiple OCR regions on one line: {line!r}",
            )

    return VerificationResult(
        passed=False,
        method="ocr_text_present",
        detail=f"{expected_substring!r} not found among {len(elements)} OCR text regions (single or per-line joined)",
    )
