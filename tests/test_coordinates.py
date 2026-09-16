"""Unit tests for visipilot.grounding.coordinates.

Covers the DPR/zoom/scroll matrix called for in Instructions.md #5 and
TESTING.md #1. Pure math, no browser required.
"""
from __future__ import annotations

import pytest

from visipilot.grounding.coordinates import (
    CoordinateError,
    bbox_to_viewport_click_point,
    is_within_viewport,
    page_css_to_viewport_css,
    screenshot_point_to_viewport_css,
    viewport_css_to_page_css,
    viewport_css_to_screenshot_point,
)
from visipilot.types import BBox, CoordinateSpace, ScreenshotMeta


def make_meta(**overrides) -> ScreenshotMeta:
    defaults = dict(
        viewport_width=1280,
        viewport_height=800,
        device_scale_factor=1.0,
        zoom_factor=1.0,
        scroll_x=0.0,
        scroll_y=0.0,
        full_page=False,
    )
    defaults.update(overrides)
    return ScreenshotMeta(**defaults)


# --- screenshot_point_to_viewport_css: DPR sweep -----------------------

@pytest.mark.parametrize("dpr", [1.0, 1.25, 1.5, 2.0])
def test_dpr_scaling_viewport_screenshot(dpr):
    meta = make_meta(device_scale_factor=dpr)
    # A point at screenshot pixel (dpr * 100, dpr * 50) must map back to
    # viewport CSS (100, 50) exactly.
    css_x, css_y = screenshot_point_to_viewport_css(dpr * 100, dpr * 50, meta)
    assert css_x == pytest.approx(100.0)
    assert css_y == pytest.approx(50.0)


@pytest.mark.parametrize("dpr", [1.0, 1.25, 1.5, 2.0])
def test_dpr_round_trip(dpr):
    meta = make_meta(device_scale_factor=dpr)
    orig = (317.0, 212.5)
    screenshot_pt = (orig[0] * dpr, orig[1] * dpr)
    css_pt = screenshot_point_to_viewport_css(*screenshot_pt, meta)
    assert css_pt[0] == pytest.approx(orig[0])
    assert css_pt[1] == pytest.approx(orig[1])
    back = viewport_css_to_screenshot_point(*css_pt, meta)
    assert back[0] == pytest.approx(screenshot_pt[0])
    assert back[1] == pytest.approx(screenshot_pt[1])


# --- zoom sweep ----------------------------------------------------------

@pytest.mark.parametrize("zoom", [1.0, 1.25, 1.5])
def test_zoom_scaling(zoom):
    meta = make_meta(device_scale_factor=1.0, zoom_factor=zoom)
    css_x, css_y = screenshot_point_to_viewport_css(zoom * 200, zoom * 100, meta)
    assert css_x == pytest.approx(200.0)
    assert css_y == pytest.approx(100.0)


def test_combined_dpr_and_zoom():
    meta = make_meta(device_scale_factor=1.5, zoom_factor=1.25)
    # effective_scale = 1.875
    css_x, css_y = screenshot_point_to_viewport_css(187.5, 375.0, meta)
    assert css_x == pytest.approx(100.0)
    assert css_y == pytest.approx(200.0)


# --- scroll offset (full-page screenshots only) --------------------------

def test_scroll_offset_ignored_for_viewport_screenshot():
    # full_page=False: a viewport-clipped screenshot already only shows
    # what's visible, so scroll offset must NOT be subtracted again.
    meta = make_meta(scroll_x=500.0, scroll_y=300.0, full_page=False)
    css_x, css_y = screenshot_point_to_viewport_css(100.0, 50.0, meta)
    assert css_x == pytest.approx(100.0)
    assert css_y == pytest.approx(50.0)


def test_scroll_offset_applied_for_full_page_screenshot():
    meta = make_meta(scroll_x=500.0, scroll_y=300.0, full_page=True)
    # Point at page-absolute (600, 350) should map to viewport-relative
    # (100, 50) once the current scroll offset is subtracted.
    css_x, css_y = screenshot_point_to_viewport_css(600.0, 350.0, meta)
    assert css_x == pytest.approx(100.0)
    assert css_y == pytest.approx(50.0)


def test_combined_dpr_zoom_and_scroll():
    meta = make_meta(
        device_scale_factor=2.0,
        zoom_factor=1.25,
        scroll_x=400.0,
        scroll_y=200.0,
        full_page=True,
    )
    scale = 2.5
    # Page-absolute point (500, 250) at this scale.
    screenshot_x = 500.0 * scale
    screenshot_y = 250.0 * scale
    css_x, css_y = screenshot_point_to_viewport_css(screenshot_x, screenshot_y, meta)
    assert css_x == pytest.approx(500.0 - 400.0)
    assert css_y == pytest.approx(250.0 - 200.0)


# --- viewport <-> page CSS -------------------------------------------------

def test_viewport_to_page_and_back():
    meta = make_meta(scroll_x=123.0, scroll_y=45.0)
    page = viewport_css_to_page_css(10.0, 20.0, meta)
    assert page == pytest.approx((133.0, 65.0))
    back = page_css_to_viewport_css(*page, meta)
    assert back == pytest.approx((10.0, 20.0))


# --- bbox_to_viewport_click_point ----------------------------------------

def test_bbox_to_viewport_click_point_center():
    meta = make_meta(device_scale_factor=2.0)
    bbox = BBox(x=100, y=200, width=40, height=20, space=CoordinateSpace.SCREENSHOT_PX)
    # center in screenshot px = (120, 210); at DPR=2 -> viewport css (60, 105)
    cx, cy = bbox_to_viewport_click_point(bbox, meta)
    assert cx == pytest.approx(60.0)
    assert cy == pytest.approx(105.0)


def test_bbox_wrong_space_raises():
    meta = make_meta()
    bbox = BBox(x=0, y=0, width=10, height=10, space=CoordinateSpace.VIEWPORT_CSS)
    with pytest.raises(CoordinateError):
        bbox_to_viewport_click_point(bbox, meta)


# --- error handling --------------------------------------------------------

def test_zero_scale_raises():
    meta = make_meta(device_scale_factor=0.0)
    with pytest.raises(CoordinateError):
        screenshot_point_to_viewport_css(10, 10, meta)


# --- is_within_viewport ----------------------------------------------------

def test_is_within_viewport_true_and_false():
    meta = make_meta(viewport_width=1280, viewport_height=800)
    assert is_within_viewport(0, 0, meta) is True
    assert is_within_viewport(1280, 800, meta) is True
    assert is_within_viewport(1281, 100, meta) is False
    assert is_within_viewport(100, -1, meta) is False
