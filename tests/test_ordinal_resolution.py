"""Real-browser integration tests for ordinal/reference resolution
(Phase C.2, "open the second result") — real Playwright, real OWLv2
detector, real EasyOCR, real click execution against real pages, scored
via DOM state (eval-only, used here only to confirm *which* real element
was actually clicked, never to produce the click itself — Instructions.md
#2).

Covers the milestone's required scenarios: the existing duplicate-result
page, first/second/third on a real results list, a missing ordinal
position, visually-reordered elements (DOM order != reading order), and
that ordinal-free ambiguity handling is unchanged.
"""
from __future__ import annotations

import pytest

from visipilot.capture.screenshot import launch_page
from visipilot.eval.dom_ground_truth import get_element_bbox
from visipilot.testserver import TestPageServer
from visipilot.tracing.pipeline import run_instruction
from visipilot.types import ActionOutcome


def _click_landed_in(click_point, bbox) -> bool:
    x, y = click_point
    return bbox.x <= x <= bbox.x2 and bbox.y <= y <= bbox.y2

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def test_server():
    with TestPageServer(port=0) as server:
        yield server


@pytest.fixture(scope="module")
def models():
    from visipilot.perception.detector import UIDetector
    from visipilot.perception.ocr import OCREngine

    return UIDetector(), OCREngine(gpu=True)


def _run(test_server, models, page_name, instruction, **kwargs):
    detector, ocr = models
    url = test_server.url(page_name)
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        record = run_instruction(page, instruction, detector, ocr, write_trace=False, **kwargs)
        results_text = page.eval_on_selector("#results", "el => el.textContent")
    return record, results_text


# --- first / second / third on a real results list -------------------------

def test_open_the_second_result_clicks_the_real_second_element(test_server, models):
    record, results_text = _run(test_server, models, "search_results_list.html", "Open the second result.")
    assert record.failure_reason is None, record.failure_reason
    assert record.action_records[-1].outcome == ActionOutcome.SUCCESS
    assert results_text == "Opened result 2"


def test_click_the_first_result(test_server, models):
    record, results_text = _run(test_server, models, "search_results_list.html", "Click the first result.")
    assert record.failure_reason is None, record.failure_reason
    assert results_text == "Opened result 1"


def test_click_the_third_result(test_server, models):
    record, results_text = _run(test_server, models, "search_results_list.html", "Click the third result.")
    assert record.failure_reason is None, record.failure_reason
    assert results_text == "Opened result 3"


def test_click_the_last_result(test_server, models):
    record, results_text = _run(test_server, models, "search_results_list.html", "Click the last result.")
    assert record.failure_reason is None, record.failure_reason
    assert results_text == "Opened result 3"


# --- missing ordinal position -----------------------------------------------

def test_missing_ordinal_position_refuses_on_a_real_page(test_server, models):
    # Only 3 results exist on this page; the 5th does not.
    record, results_text = _run(test_server, models, "search_results_list.html", "Click the fifth result.")
    assert record.failure_reason is not None
    assert record.action_records[-1].outcome == ActionOutcome.FAILED_ORDINAL_OUT_OF_RANGE
    assert results_text == ""  # nothing was clicked


# --- reordered elements: visual order must win over DOM order --------------

def test_ordinal_uses_visual_reading_order_not_dom_source_order(test_server, models):
    # search_results_reordered.html's DOM order is Cherries, Bananas,
    # Apples; its rendered (column-reverse) order, top to bottom, is
    # Apples, Bananas, Cherries. "The first result" must resolve to the
    # visually topmost item (Apples) -- pixels-first perception never
    # sees DOM order at all, so there'd be no way for it to prefer
    # Cherries even if it wanted to; this proves the ordering actually
    # used is bbox position, not an accidental DOM-order leak somewhere
    # in fusion/OCR ordering.
    record, results_text = _run(test_server, models, "search_results_reordered.html", "Click the first result.")
    assert record.failure_reason is None, record.failure_reason
    assert results_text == "Opened result: Apples"


def test_ordinal_third_on_reordered_page_is_the_visually_last_item(test_server, models):
    record, results_text = _run(test_server, models, "search_results_reordered.html", "Click the third result.")
    assert record.failure_reason is None, record.failure_reason
    assert results_text == "Opened result: Cherries"


# --- the existing duplicate-result page: real evaluation case --------------

def test_duplicate_page_ordinal_free_reference_still_refuses_unchanged(test_server, models):
    # Regression check required by this milestone: an instruction with
    # no ordinal word, against the page that already produced a genuine
    # 3-way tie, must behave exactly as before ordinal support existed.
    record, results_text = _run(test_server, models, "search_duplicate.html",
                                "Find the search box, type Python, and click Search.")
    assert record.failure_reason is not None
    assert record.action_records[-1].outcome == ActionOutcome.FAILED_AMBIGUOUS
    assert results_text == ""


def test_duplicate_page_ordinal_click_lands_on_the_real_second_button(test_server, models):
    # Real evaluation case required by this milestone: bare "Search" ties
    # cleanly on exactly the two real buttons (both score an exact-match
    # bonus that the panel headings' "Site search"/"Product search" text
    # doesn't get), so "the second Search" is a clean, honest ordinal
    # demonstration on the page this project already uses for duplicate-
    # element ambiguity. Scored against real DOM ground truth (eval-only)
    # rather than just trusting the outcome, since a click landing near
    # but not on the intended element would still report SUCCESS.
    detector, ocr = models
    url = test_server.url("search_duplicate.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        truth_btn2 = get_element_bbox(page, "#search-btn-2")
        record = run_instruction(page, "Click the second Search.", detector, ocr, write_trace=False)

    assert record.failure_reason is None, record.failure_reason
    click_point = record.action_records[0].click_point
    assert click_point is not None
    assert _click_landed_in(click_point, truth_btn2)


def test_duplicate_page_ordinal_click_first_lands_on_the_real_first_button(test_server, models):
    detector, ocr = models
    url = test_server.url("search_duplicate.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        truth_btn1 = get_element_bbox(page, "#search-btn")
        record = run_instruction(page, "Click the first Search.", detector, ocr, write_trace=False)

    assert record.failure_reason is None, record.failure_reason
    click_point = record.action_records[0].click_point
    assert click_point is not None
    assert _click_landed_in(click_point, truth_btn1)


def test_duplicate_page_missing_ordinal_position_refuses(test_server, models):
    # Only 2 "Search" buttons exist on this page; a 3rd reference must
    # safely refuse, not fall back to either real one.
    record, _ = _run(test_server, models, "search_duplicate.html", "Click the third Search.")
    assert record.failure_reason is not None
    assert record.action_records[-1].outcome == ActionOutcome.FAILED_ORDINAL_OUT_OF_RANGE
