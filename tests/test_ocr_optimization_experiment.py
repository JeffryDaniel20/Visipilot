"""Tests for visipilot.eval.ocr_optimization_experiment (Phase D.2).

These lock in the three real negative findings so a future dependency
upgrade (EasyOCR, onnxruntime-gpu) or environment change (a system CUDA
Toolkit getting installed for an unrelated reason) is caught rather than
silently invalidating this milestone's documented conclusion — a passing
test here should prompt re-reading implementation-plan.md D.2, not be
quietly ignored.

`onnxruntime-gpu`/`onnx`/`onnxscript` were installed locally only for
this experiment (implementation-plan.md D.2), not added to
requirements.txt, since no optimization was adopted — the ONNX-related
test skips gracefully if they're not present in a given environment.
"""
from __future__ import annotations

import pytest

from visipilot.capture.screenshot import capture_screenshot, launch_page
from visipilot.eval.ocr_optimization_experiment import (
    check_fp16_detector_crashes,
    check_fp16_recognizer_accuracy_and_speed,
    check_onnxruntime_cuda_ep,
    check_quantize_is_noop_on_gpu,
)
from visipilot.eval.page_suite import TESTPAGES_DIR
from visipilot.perception.ocr import OCREngine
from visipilot.testserver import TestPageServer

pytestmark = pytest.mark.integration


def test_quantize_flag_remains_a_noop_on_gpu():
    result = check_quantize_is_noop_on_gpu()
    assert result["dtypes_identical"] is True
    assert result["detector_type_matches"] is True
    assert result["recognizer_type_matches"] is True


@pytest.fixture(scope="module")
def basic_page_screenshot_path():
    with TestPageServer(directory=str(TESTPAGES_DIR), port=0) as server:
        url = server.url("search_basic.html")
        with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
            shot = capture_screenshot(page)
            yield shot.image_path


def test_fp16_detector_still_crashes(basic_page_screenshot_path):
    ocr = OCREngine(gpu=True)
    result = check_fp16_detector_crashes(ocr, basic_page_screenshot_path)
    assert result["crashed"] is True, (
        "FP16 detector no longer crashes -- this contradicts implementation-plan.md D.2's "
        "documented finding; re-investigate whether FP16 is now viable before assuming "
        "this test is simply wrong"
    )
    assert result["exception_type"] == "error"  # cv2.error


def test_fp16_recognizer_preserves_accuracy_but_not_speed(basic_page_screenshot_path):
    ocr = OCREngine(gpu=True)
    result = check_fp16_recognizer_accuracy_and_speed(ocr, basic_page_screenshot_path, runs=3)
    assert result["same_text_recognized"] is True
    assert result["max_confidence_delta"] < 0.05
    # Not asserting a specific negative speedup number here (real timing
    # varies run to run) -- the milestone's real, repeated-page finding
    # is documented in implementation-plan.md D.2; this test only checks
    # accuracy is preserved, which is the more stable, meaningful property.


def test_onnxruntime_cuda_ep_still_unavailable():
    pytest.importorskip("onnxruntime", reason="onnxruntime-gpu installed locally only for the D.2 experiment")
    result = check_onnxruntime_cuda_ep()
    assert result["installed"] is True
    assert result["cuda_ep_actually_used"] is False, (
        "CUDAExecutionProvider now works in this environment -- this contradicts "
        "implementation-plan.md D.2's documented finding (missing CUDA 13.x/cuDNN 9.x); "
        "if a system CUDA Toolkit was installed, re-evaluate ONNX Runtime as an OCR "
        "optimization candidate rather than assuming this test is simply wrong"
    )
