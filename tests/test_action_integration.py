"""Integration test: real screenshot -> real OWLv2/EasyOCR -> real
semantic builder -> real instruction parsing -> real matcher -> real
Playwright click/type actions -> real pixels-first verification,
cross-checked against eval-only DOM ground truth/verification.

This is the strongest evidence available for A.5 at this milestone. It
also honestly demonstrates, rather than hides, the one place the system
currently refuses to act: the real vertical-slice instruction's CLICK
step is a genuine tie between the real input and the real button's OCR
text (see implementation-plan.md A.4), so the full instruction halts
there by design instead of guessing.
"""
from __future__ import annotations

import pytest

from visipilot.action.executor import AmbiguousTargetError, resolve_single_candidate
from visipilot.action.runner import run_steps
from visipilot.action.verification import verify_text_present
from visipilot.capture.screenshot import capture_screenshot, launch_page
from visipilot.eval.dom_ground_truth import get_element_bbox
from visipilot.eval.dom_verification import dom_text_present
from visipilot.grounding.coordinates import screenshot_point_to_viewport_css
from visipilot.semantic.builder import build_semantic_state
from visipilot.target_selection.instruction_parser import parse_instruction
from visipilot.target_selection.matcher import match_target
from visipilot.testserver import TestPageServer
from visipilot.types import ActionOutcome

pytestmark = pytest.mark.integration

VERTICAL_SLICE_INSTRUCTION = "Find the search box, type Python, and click Search."


@pytest.fixture(scope="module")
def test_server():
    with TestPageServer(port=0) as server:
        yield server


@pytest.fixture(scope="module")
def models():
    from visipilot.perception.detector import UIDetector
    from visipilot.perception.ocr import OCREngine

    return UIDetector(), OCREngine(gpu=True)


def _build_state(page, models):
    detector, ocr = models
    shot = capture_screenshot(page)
    det_elements = detector.detect(shot.image_path)
    ocr_elements = ocr.read(shot.image_path)
    return build_semantic_state(shot, det_elements, ocr_elements)


def test_click_subscribe_real_click_lands_on_real_button(test_server, models):
    """The one real actionable-and-unambiguous target on this page:
    "Subscribe" appears nowhere else, so this exercises a full, real,
    trusted-input click end to end.
    """
    url = test_server.url("search_basic.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        state = _build_state(page, models)
        truth = get_element_bbox(page, "#subscribe-btn")

        steps = parse_instruction("Click Subscribe.")
        records = run_steps(page, steps, state)

    assert len(records) == 1
    assert records[0].outcome == ActionOutcome.SUCCESS
    click_x, click_y = records[0].click_point
    assert truth.x <= click_x <= truth.x2
    assert truth.y <= click_y <= truth.y2


def test_find_and_type_search_box_then_verify_pixels_first(test_server, models):
    """Exercises FIND + TYPE for real, then verifies the typed text is
    visible in a fresh screenshot via OCR — no DOM read for the
    verification itself.
    """
    detector, ocr = models
    url = test_server.url("search_basic.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        state = _build_state(page, models)

        steps = parse_instruction("Find the search box, type Python.")
        records = run_steps(page, steps, state)
        assert [r.outcome for r in records] == [ActionOutcome.SUCCESS, ActionOutcome.SUCCESS]

        after_shot = capture_screenshot(page)

    result = verify_text_present(ocr, after_shot.image_path, "Python")
    assert result.passed, result.detail


def test_full_vertical_slice_instruction_halts_honestly_at_click(test_server, models):
    """The real, current, documented outcome: FIND and TYPE succeed,
    CLICK is a genuine tie (the real input's "Search:" and the real
    button's "Search" score identically for the bare phrase "Search" —
    see implementation-plan.md A.4) and the runner refuses to guess.
    """
    url = test_server.url("search_basic.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        state = _build_state(page, models)
        steps = parse_instruction(VERTICAL_SLICE_INSTRUCTION)
        records = run_steps(page, steps, state)

    assert [r.action for r in records] == ["find", "type", "click"]
    assert records[0].outcome == ActionOutcome.SUCCESS
    assert records[1].outcome == ActionOutcome.SUCCESS
    assert records[2].outcome == ActionOutcome.FAILED_AMBIGUOUS

    # Confirm it really is the matcher tie causing this, not some other
    # failure mode, by reproducing it directly against match_target.
    candidates = match_target("Search", state)
    with pytest.raises(AmbiguousTargetError):
        resolve_single_candidate(candidates)


def test_pixels_first_and_dom_verification_agree_after_a_real_search(test_server, models):
    """Cross-checks the pixels-first verifier against the eval-only DOM
    verifier on the same real state change — DOM access here is purely
    test scaffolding (triggering the click directly, the same way
    dom_ground_truth.py's tests already use the DOM for setup/ground
    truth), never part of the runtime verification path itself.
    """
    detector, ocr = models
    url = test_server.url("search_basic.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        page.fill("#search-box", "Python")
        page.click("#search-btn")  # DOM-driven, test setup only
        after_shot = capture_screenshot(page)
        dom_result = dom_text_present(page, "#results", "Results for: Python")

    pixels_result = verify_text_present(ocr, after_shot.image_path, "Results for: Python")

    assert dom_result is True
    assert pixels_result.passed is True
