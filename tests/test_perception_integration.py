"""Integration test: real OWLv2 detector + real EasyOCR + fusion, against
a real screenshot of the controlled test page.

Requires torch/transformers/easyocr installed and a GPU (or falls back to
CPU automatically if torch.cuda.is_available() is False). Marked as an
integration test since it downloads/loads real model weights and is slow
relative to the unit suite.
"""
from __future__ import annotations

import pytest

from visipilot.capture.screenshot import capture_screenshot, launch_page
from visipilot.eval.dom_ground_truth import get_element_bbox
from visipilot.perception.fusion import fuse
from visipilot.testserver import TestPageServer

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def test_server():
    with TestPageServer(port=0) as server:
        yield server


@pytest.fixture(scope="module")
def screenshot_and_ground_truth(test_server):
    url = test_server.url("search_basic.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        shot = capture_screenshot(page)
        truth = {
            "search_box": get_element_bbox(page, "#search-box"),
            "search_btn": get_element_bbox(page, "#search-btn"),
        }
    return shot, truth


@pytest.fixture(scope="module")
def detector():
    from visipilot.perception.detector import UIDetector

    return UIDetector()


@pytest.fixture(scope="module")
def ocr_engine():
    from visipilot.perception.ocr import OCREngine

    return OCREngine(gpu=True)


def test_detector_produces_elements(screenshot_and_ground_truth, detector):
    shot, _ = screenshot_and_ground_truth
    elements = detector.detect(shot.image_path)
    assert isinstance(elements, list)
    for el in elements:
        assert el.bbox.width > 0
        assert el.bbox.height > 0
        assert 0.0 <= el.confidence <= 1.0


def test_ocr_finds_search_button_text(screenshot_and_ground_truth, ocr_engine):
    shot, _ = screenshot_and_ground_truth
    elements = ocr_engine.read(shot.image_path)
    texts = [el.text for el in elements if el.text]
    assert any("search" in t.lower() for t in texts), f"expected 'Search' text among OCR results, got: {texts}"


def test_fusion_pipeline_end_to_end(screenshot_and_ground_truth, detector, ocr_engine):
    shot, truth = screenshot_and_ground_truth
    det_elements = detector.detect(shot.image_path)
    ocr_elements = ocr_engine.read(shot.image_path)
    fused = fuse(det_elements, ocr_elements)

    # Every OCR/detector element must be preserved (fusion only merges,
    # never drops silently) — the fused count is at most the sum and at
    # least the max of the two inputs.
    assert len(fused) <= len(det_elements) + len(ocr_elements)
    assert len(fused) >= max(len(det_elements), len(ocr_elements))
