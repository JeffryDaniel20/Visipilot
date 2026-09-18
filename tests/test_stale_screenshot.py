"""Real-browser tests for Instructions.md #7's stale-screenshot rule:
`visipilot.capture.screenshot.screenshot_has_changed()` and
`visipilot.action.runner.run_steps()`'s refusal when the live page no
longer matches the screenshot perception ran against.

Uses real Playwright + the local test-page server (same pattern as
`test_capture_integration.py`), not mocks — staleness detection is
fundamentally about comparing real rendered pixels over real time, which
a mock can't meaningfully stand in for.
"""
from __future__ import annotations

import pytest

from visipilot.action.runner import run_steps
from visipilot.capture.screenshot import capture_screenshot, launch_page, screenshot_has_changed
from visipilot.eval.dom_ground_truth import get_element_bbox
from visipilot.target_selection.instruction_parser import parse_instruction
from visipilot.testserver import TestPageServer
from visipilot.types import (
    ActionOutcome,
    BBox,
    CoordinateSpace,
    ElementSource,
    ElementType,
    SemanticUIState,
    UIElement,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def test_server():
    with TestPageServer(port=0) as server:
        yield server


def _real_state_for_dynamic_page(page, shot):
    """Build a SemanticUIState from real DOM ground truth (eval-only,
    used here only to get an accurate bbox without depending on the
    detector/OCR's own accuracy — this test is about staleness
    detection, not perception quality, which has its own dedicated
    tests elsewhere).

    Deliberately includes ONLY the button, not the search input: the
    test below only ever runs a "Click Search." instruction (no FIND/
    TYPE), and an earlier version of this fixture included both,
    letting `match_target`'s real fill-color tie-breaker
    (`_fill_bonus`) decide between them — a real signal, but one this
    test doesn't need to exercise, and sampling real pixel colors made
    the test's outcome sensitive to unrelated system load when run deep
    inside the full suite (found via a real, reproducible full-suite
    failure, not assumed). A single unambiguous candidate removes that
    dependency entirely without weakening what this test actually checks.
    """
    btn = get_element_bbox(page, "#search-btn")

    def to_screenshot_bbox(b):
        return BBox(x=b.x, y=b.y, width=b.width, height=b.height, space=CoordinateSpace.SCREENSHOT_PX)

    elements = [
        UIElement(id="btn", type=ElementType.BUTTON, bbox=to_screenshot_bbox(btn), text="Search", confidence=0.9, interactable=True, source=ElementSource.FUSION),
    ]
    return SemanticUIState(elements=elements, screenshot=shot)


def test_screenshot_has_changed_false_positive_free_on_static_page(test_server):
    # No wait, no interaction -- an unchanged page must never be flagged
    # stale (a real false-positive would halt every normal run).
    url = test_server.url("search_basic.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        shot = capture_screenshot(page)
        assert screenshot_has_changed(page, shot) is False


def test_screenshot_has_changed_detects_real_late_banner(test_server):
    # search_dynamic.html's banner appears ~600ms after load, shifting
    # the search row down -- a real, external layout change.
    url = test_server.url("search_dynamic.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        shot = capture_screenshot(page)
        page.wait_for_timeout(900)  # let the banner actually appear
        assert screenshot_has_changed(page, shot) is True


def test_run_steps_refuses_click_when_page_changed_since_perception(test_server):
    # End-to-end: perception "sees" the pre-banner layout, the banner
    # appears before the CLICK step executes, and the runner must refuse
    # rather than click coordinates that (in a real, taller-banner case)
    # would now land on the wrong element.
    url = test_server.url("search_dynamic.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        shot = capture_screenshot(page)
        state = _real_state_for_dynamic_page(page, shot)
        page.wait_for_timeout(900)  # page mutates here, same as a real slow-detector window would allow

        steps = parse_instruction("Click Search.")
        records = run_steps(page, steps, state)

        assert len(records) == 1
        assert records[0].outcome == ActionOutcome.FAILED_STALE_SCREENSHOT
        assert records[0].error_message is not None and "changed" in records[0].error_message


def test_run_steps_succeeds_normally_when_page_has_not_changed(test_server):
    # Regression guard for the fix itself: a normal, real multi-step run
    # (FIND -> TYPE -> CLICK) on a page that does NOT introduce external
    # changes must still succeed -- the TYPE step's own echoed text must
    # not be mistaken for staleness (the bug found and fixed this
    # milestone: comparing every step against the *original* screenshot
    # made the CLICK step always look stale after TYPE legitimately
    # changed the input's rendered content).
    url = test_server.url("search_basic.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        shot = capture_screenshot(page)
        input_box = get_element_bbox(page, "#search-box")
        btn_box = get_element_bbox(page, "#search-btn")

        def to_screenshot_bbox(b):
            return BBox(x=b.x, y=b.y, width=b.width, height=b.height, space=CoordinateSpace.SCREENSHOT_PX)

        elements = [
            UIElement(id="input", type=ElementType.TEXT_INPUT, bbox=to_screenshot_bbox(input_box), text="Search:", confidence=0.9, interactable=True, source=ElementSource.FUSION),
            UIElement(id="btn", type=ElementType.BUTTON, bbox=to_screenshot_bbox(btn_box), text="Search", confidence=0.9, interactable=True, source=ElementSource.FUSION),
        ]
        state = SemanticUIState(elements=elements, screenshot=shot)

        steps = parse_instruction("Find the search box, type Python, and click Search.")
        records = run_steps(page, steps, state)

        assert [r.outcome for r in records] == [ActionOutcome.SUCCESS, ActionOutcome.SUCCESS, ActionOutcome.SUCCESS]
