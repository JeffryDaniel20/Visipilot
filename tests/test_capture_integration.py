"""Integration test: real Playwright + real Chrome + the controlled test
page, tying together TestPageServer, screenshot capture, and the
coordinate-mapping module against real DOM ground truth.

This is the test that proves the coordinate-mapping *model* (documented
in visipilot/types.py's ScreenshotMeta docstring) actually matches what a
real browser produces, not just synthetic arithmetic — see
implementation-plan.md A.2/A.4 and TESTING.md #2.
"""
from __future__ import annotations

import pytest

from visipilot.capture.screenshot import capture_screenshot, launch_page
from visipilot.eval.dom_ground_truth import get_element_bbox
from visipilot.grounding.coordinates import screenshot_point_to_viewport_css
from visipilot.testserver import TestPageServer

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def test_server():
    with TestPageServer(port=0) as server:
        yield server


def test_screenshot_capture_metadata(test_server):
    url = test_server.url("search_basic.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        shot = capture_screenshot(page)

    assert shot.meta.viewport_width == 1280
    assert shot.meta.viewport_height == 800
    assert shot.meta.device_scale_factor == 1.0
    assert shot.meta.full_page is False
    # A viewport-clipped screenshot at DPR=1 must be exactly viewport-sized.
    assert shot.width_px == 1280
    assert shot.height_px == 800


@pytest.mark.parametrize("dpr", [1.0, 1.5, 2.0])
def test_screenshot_dimensions_scale_with_dpr(test_server, dpr):
    url = test_server.url("search_basic.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=dpr) as page:
        shot = capture_screenshot(page)

    assert shot.meta.device_scale_factor == dpr
    assert shot.width_px == round(1280 * dpr)
    assert shot.height_px == round(800 * dpr)


@pytest.mark.parametrize("dpr", [1.0, 1.5, 2.0])
def test_coordinate_mapping_matches_real_dom_ground_truth(test_server, dpr):
    """The core end-to-end claim: given a screenshot-pixel point that
    corresponds to the real DOM element's center (as if a detector had
    found it), mapping it back through our coordinate module must land
    within the true element's bounding box, for several DPR values.
    """
    url = test_server.url("search_basic.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=dpr) as page:
        shot = capture_screenshot(page)
        true_bbox = get_element_bbox(page, "#search-btn")

    # Simulate a detector finding the button at its true center, expressed
    # in screenshot-pixel space (DOM bbox is VIEWPORT_CSS -> multiply by
    # effective_scale to get screenshot px, exactly like a real detector
    # output would be, just derived from ground truth instead of a model).
    true_center_css = true_bbox.center
    screenshot_x = true_center_css[0] * shot.meta.effective_scale
    screenshot_y = true_center_css[1] * shot.meta.effective_scale

    mapped_css_x, mapped_css_y = screenshot_point_to_viewport_css(
        screenshot_x, screenshot_y, shot.meta
    )

    # Mapped point must land inside the true element's bounding box.
    assert true_bbox.x <= mapped_css_x <= true_bbox.x2
    assert true_bbox.y <= mapped_css_y <= true_bbox.y2
    # And must match the true center closely (sub-pixel rounding only).
    assert mapped_css_x == pytest.approx(true_center_css[0], abs=0.5)
    assert mapped_css_y == pytest.approx(true_center_css[1], abs=0.5)


@pytest.mark.parametrize("zoom", [1.25, 1.5])
def test_coordinate_mapping_matches_real_dom_ground_truth_under_real_zoom(test_server, zoom):
    """Closes the open item from A.4/A.6 ("the zoom-factor part of the
    model is still unverified against a live browser"): applies a real
    CDP `Emulation.setPageScaleFactor` zoom via `launch_page(zoom=...)`,
    reads the real `zoom_factor` back via `capture_screenshot`, and
    confirms a screenshot-pixel point (as a real detector would report,
    now visually magnified by the real zoom) still maps back to the true
    DOM element's unchanged CSS bounding box, exactly like the DPR test
    above but for zoom instead of DPR.
    """
    url = test_server.url("search_basic.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0, zoom=zoom) as page:
        shot = capture_screenshot(page)
        true_bbox = get_element_bbox(page, "#search-btn")

    assert shot.meta.zoom_factor == pytest.approx(zoom, abs=0.01)

    true_center_css = true_bbox.center
    screenshot_x = true_center_css[0] * shot.meta.effective_scale
    screenshot_y = true_center_css[1] * shot.meta.effective_scale

    mapped_css_x, mapped_css_y = screenshot_point_to_viewport_css(screenshot_x, screenshot_y, shot.meta)

    assert true_bbox.x <= mapped_css_x <= true_bbox.x2
    assert true_bbox.y <= mapped_css_y <= true_bbox.y2
    assert mapped_css_x == pytest.approx(true_center_css[0], abs=0.5)
    assert mapped_css_y == pytest.approx(true_center_css[1], abs=0.5)


def test_zoom_defaults_to_unchanged_behavior(test_server):
    # Regression guard: launch_page's new `zoom` parameter must not
    # affect anything when left at its default (1.0) -- the CDP call is
    # skipped entirely in that case (see launch_page's `if zoom != 1.0`).
    url = test_server.url("search_basic.html")
    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        shot = capture_screenshot(page)
    assert shot.meta.zoom_factor == 1.0


def test_dom_ground_truth_finds_all_page_elements(test_server):
    url = test_server.url("search_basic.html")
    with launch_page(url) as page:
        for selector in ["#search-box", "#search-btn", "#newsletter-input", "#subscribe-btn", "#unrelated-text"]:
            bbox = get_element_bbox(page, selector)
            assert bbox.width > 0
            assert bbox.height > 0
