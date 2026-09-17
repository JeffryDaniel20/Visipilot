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
