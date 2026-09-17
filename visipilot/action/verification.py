"""Verification: confirms an action had its expected effect via
pixels-first visual/OCR diffing on a fresh screenshot — never via the
DOM at runtime (Instructions.md #2/#5). DOM-based confirmation is
eval-only and lives in visipilot/eval/; nothing in this module imports
it.
"""
from __future__ import annotations

from pathlib import Path

from visipilot.perception.ocr import OCREngine
from visipilot.types import VerificationResult


def verify_text_present(ocr: OCREngine, screenshot_image_path: str | Path, expected_substring: str) -> VerificationResult:
    """Re-runs OCR on a screenshot taken after an action and checks
    whether `expected_substring` appears anywhere in the read text
    (case-insensitive). This is the general-purpose pixels-first check
    for "did the page visibly show the result I expected" — e.g. a
    search-results message appearing after a search action.
    """
    elements = ocr.read(screenshot_image_path)
    needle = expected_substring.lower()
    for el in elements:
        if el.text and needle in el.text.lower():
            return VerificationResult(
                passed=True,
                method="ocr_text_present",
                detail=f"found {expected_substring!r} in OCR text {el.text!r}",
            )
    return VerificationResult(
        passed=False,
        method="ocr_text_present",
        detail=f"{expected_substring!r} not found among {len(elements)} OCR text regions",
    )
