"""Integration test: real screenshot -> real OWLv2 detector -> real
EasyOCR -> real Semantic UI Representation Builder -> real instruction
parsing -> real target-selection matching, scored against real DOM
ground truth (eval-only, used here only to check results, never to
produce them — see the pixels-first rule in Instructions.md #2).

This is the strongest evidence available for A.3/A.4 at this milestone:
it proves the whole chain works on real data, not just synthetic
fixtures, and it's what surfaced the real distractor-text bug fixed in
visipilot/target_selection/matcher.py (see implementation-plan.md).
"""
from __future__ import annotations

import pytest

from visipilot.capture.screenshot import capture_screenshot, launch_page
from visipilot.eval.dom_ground_truth import get_element_bbox
from visipilot.grounding.coordinates import screenshot_point_to_viewport_css
from visipilot.semantic.builder import build_semantic_state
from visipilot.target_selection.instruction_parser import ActionKind, parse_instruction
from visipilot.target_selection.matcher import match_target
from visipilot.testserver import TestPageServer

pytestmark = pytest.mark.integration

VERTICAL_SLICE_INSTRUCTION = "Find the search box, type Python, and click Search."


@pytest.fixture(scope="module")
def test_server():
    with TestPageServer(port=0) as server:
        yield server


@pytest.fixture(scope="module")
def pipeline_output(test_server):
    url = test_server.url("search_basic.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        shot = capture_screenshot(page)
        truth = {
            "search_box": get_element_bbox(page, "#search-box"),
            "search_btn": get_element_bbox(page, "#search-btn"),
        }

    from visipilot.perception.detector import UIDetector
    from visipilot.perception.ocr import OCREngine

    detector = UIDetector()
    ocr = OCREngine(gpu=True)
    det_elements = detector.detect(shot.image_path)
    ocr_elements = ocr.read(shot.image_path)
    state = build_semantic_state(shot, det_elements, ocr_elements)
    return state, truth


def _candidate_hits_truth(candidate_bbox, truth_bbox, meta) -> bool:
    """True if the candidate's center, mapped into viewport CSS space,
    lands inside the true element's bounding box.
    """
    cx, cy = candidate_bbox.center
    vx, vy = screenshot_point_to_viewport_css(cx, cy, meta)
    return truth_bbox.x <= vx <= truth_bbox.x2 and truth_bbox.y <= vy <= truth_bbox.y2


def test_instruction_parses_into_three_steps():
    steps = parse_instruction(VERTICAL_SLICE_INSTRUCTION)
    assert [s.action for s in steps] == [ActionKind.FIND, ActionKind.TYPE, ActionKind.CLICK]


def test_distractor_paragraph_never_ranks_first(pipeline_output):
    """Regression coverage for the real bug found during development:
    the test page's own distractor paragraph text literally contains the
    words "search box", and without the matcher's length penalty it
    out-scored the real search input.
    """
    state, _truth = pipeline_output
    candidates = match_target("search box", state, top_k=5)
    assert candidates, "expected at least one candidate for 'search box'"
    top_text = (candidates[0].element.text or "")
    assert "distractor" not in top_text.lower()
    assert len(top_text.split()) <= 6, f"top candidate's text looks like a paragraph, not a UI label: {top_text!r}"


def test_click_search_top_candidates_include_the_real_button(pipeline_output):
    state, truth = pipeline_output
    meta = state.screenshot.meta
    candidates = match_target("Search", state, top_k=5)
    assert candidates

    hits = [c for c in candidates if _candidate_hits_truth(c.element.bbox, truth["search_btn"], meta)]
    assert hits, (
        "expected the real Search button to appear among top candidates for 'Search'; "
        f"got: {[(c.element.id, c.element.text, c.score) for c in candidates]}"
    )


def test_find_search_box_top_candidates_include_the_real_input(pipeline_output):
    """Honest expectation, not an idealized one: with the current weak
    detector type signal and OCR text alone, "the search box" and the
    Search button's own OCR text ("Search:" vs "Search") are genuinely
    hard to tell apart by text alone — both contain the token "search"
    and EasyOCR misread the input's "..." placeholder ellipsis as ":".
    The matcher is expected to surface the real search input among its
    top candidates (it does), not necessarily rank it uniquely first —
    forcing a false disambiguation would be worse than an honest tie.
    """
    state, truth = pipeline_output
    meta = state.screenshot.meta
    candidates = match_target("search box", state, top_k=5)
    assert candidates

    hits = [c for c in candidates if _candidate_hits_truth(c.element.bbox, truth["search_box"], meta)]
    assert hits, (
        "expected the real search input to appear among top candidates for 'search box'; "
        f"got: {[(c.element.id, c.element.text, c.score) for c in candidates]}"
    )
