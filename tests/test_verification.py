"""Unit tests for visipilot.action.verification — a stub OCR engine
(duck-typed to OCREngine's .read() interface), no real model needed.
"""
from __future__ import annotations

from visipilot.action.verification import verify_text_present
from visipilot.types import BBox, CoordinateSpace, ElementSource, ElementType, UIElement


class StubOCR:
    def __init__(self, texts: list[str]):
        self._texts = texts

    def read(self, image_path):
        return [
            UIElement(
                id=f"ocr-{i}",
                type=ElementType.TEXT,
                bbox=BBox(x=0, y=i * 20, width=100, height=15, space=CoordinateSpace.SCREENSHOT_PX),
                text=t,
                confidence=0.95,
                interactable=False,
                source=ElementSource.OCR,
            )
            for i, t in enumerate(self._texts)
        ]


def test_verify_text_present_found():
    ocr = StubOCR(["VisiPilot Demo", "Results for: Python", "Search"])
    result = verify_text_present(ocr, "fake.png", "Results for: Python")
    assert result.passed is True
    assert result.method == "ocr_text_present"


def test_verify_text_present_case_insensitive():
    ocr = StubOCR(["RESULTS FOR: PYTHON"])
    result = verify_text_present(ocr, "fake.png", "results for: python")
    assert result.passed is True


def test_verify_text_present_not_found():
    ocr = StubOCR(["VisiPilot Demo", "Subscribe"])
    result = verify_text_present(ocr, "fake.png", "Results for: Python")
    assert result.passed is False


def test_verify_text_present_empty_ocr():
    ocr = StubOCR([])
    result = verify_text_present(ocr, "fake.png", "anything")
    assert result.passed is False


class StubOCRWithBoxes:
    """Lets a test place OCR fragments at explicit (x, y, width, height)
    — needed to reproduce the real fragmentation scenario found in
    Phase B testing, where two fragments' precise vertical centers
    matter (see visipilot/action/verification.py's _reading_order_text).
    """

    def __init__(self, boxes: list[tuple[str, float, float, float, float]]):
        self._boxes = boxes  # (text, x, y, width, height)

    def read(self, image_path):
        return [
            UIElement(
                id=f"ocr-{i}",
                type=ElementType.TEXT,
                bbox=BBox(x=x, y=y, width=w, height=h, space=CoordinateSpace.SCREENSHOT_PX),
                text=text,
                confidence=0.95,
                interactable=False,
                source=ElementSource.OCR,
            )
            for i, (text, x, y, w, h) in enumerate(self._boxes)
        ]


def test_verify_text_present_finds_text_split_across_two_regions():
    # Reproduces the exact real case found on the Bootstrap-style test
    # page: "Results for:" and "Python" as two separate OCR regions with
    # matching vertical centers (~349) but different y-tops.
    ocr = StubOCRWithBoxes([
        ("Results for:", 49, 341, 80, 16),  # center = 349
        ("Python", 126, 335, 59, 28),         # center = 349
    ])
    result = verify_text_present(ocr, "fake.png", "Results for: Python")
    assert result.passed is True
    assert result.method == "ocr_text_present_joined"


def test_verify_text_present_split_fragments_out_of_reading_order_still_found():
    # x-order determines join order within a line, not the order the
    # OCR engine happened to return them in.
    ocr = StubOCRWithBoxes([
        ("Python", 126, 335, 59, 28),
        ("Results for:", 49, 341, 80, 16),
    ])
    result = verify_text_present(ocr, "fake.png", "Results for: Python")
    assert result.passed is True


def test_verify_text_present_does_not_merge_separate_lines():
    # Two fragments on genuinely different lines (large vertical-center
    # gap) must NOT be concatenated into a false match.
    ocr = StubOCRWithBoxes([
        ("Results for:", 49, 100, 80, 16),
        ("Python", 49, 400, 59, 20),
    ])
    result = verify_text_present(ocr, "fake.png", "Results for: Python")
    assert result.passed is False
