"""Deterministic coordinate mapping between screenshot pixels and the CSS
pixel spaces Playwright's action APIs understand.

This module is pure (no I/O, no Playwright dependency) so it can be
exhaustively unit tested independent of a running browser — see
Instructions.md #5 ("Coordinate mapping ... must have dedicated tests
before it is trusted anywhere else").
"""
from __future__ import annotations

from visipilot.types import BBox, CoordinateSpace, ScreenshotMeta


class CoordinateError(ValueError):
    """Raised when a coordinate transform is given invalid inputs."""


def screenshot_point_to_viewport_css(
    x: float, y: float, meta: ScreenshotMeta
) -> tuple[float, float]:
    """Convert a point in screenshot-pixel space to viewport CSS pixels.

    If the screenshot was taken full-page (covers the whole scrollable
    page, not just the visible viewport), the current scroll offset is
    subtracted so the result is relative to what's currently visible.
    """
    scale = meta.effective_scale
    if scale <= 0:
        raise CoordinateError(f"invalid effective_scale={scale!r}")

    css_x = x / scale
    css_y = y / scale

    if meta.full_page:
        css_x -= meta.scroll_x
        css_y -= meta.scroll_y

    return css_x, css_y


def viewport_css_to_screenshot_point(
    x: float, y: float, meta: ScreenshotMeta
) -> tuple[float, float]:
    """Inverse of screenshot_point_to_viewport_css."""
    scale = meta.effective_scale
    if scale <= 0:
        raise CoordinateError(f"invalid effective_scale={scale!r}")

    if meta.full_page:
        x = x + meta.scroll_x
        y = y + meta.scroll_y

    return x * scale, y * scale


def viewport_css_to_page_css(
    x: float, y: float, meta: ScreenshotMeta
) -> tuple[float, float]:
    return x + meta.scroll_x, y + meta.scroll_y


def page_css_to_viewport_css(
    x: float, y: float, meta: ScreenshotMeta
) -> tuple[float, float]:
    return x - meta.scroll_x, y - meta.scroll_y


def bbox_to_viewport_click_point(
    bbox: BBox, meta: ScreenshotMeta
) -> tuple[float, float]:
    """Map a detected element's bbox (in screenshot-pixel space) to a
    single click point in viewport CSS pixels, i.e. what Playwright's
    ``mouse.click`` / ``page.click`` expects.
    """
    if bbox.space != CoordinateSpace.SCREENSHOT_PX:
        raise CoordinateError(
            f"expected a bbox in SCREENSHOT_PX space, got {bbox.space!r}"
        )
    cx, cy = bbox.center
    return screenshot_point_to_viewport_css(cx, cy, meta)


def is_within_viewport(x: float, y: float, meta: ScreenshotMeta) -> bool:
    """Sanity check: does a VIEWPORT_CSS point actually land inside the
    visible viewport? A grounding result that fails this check should
    never be clicked blindly — see Instructions.md #7 (failure handling).
    """
    return 0 <= x <= meta.viewport_width and 0 <= y <= meta.viewport_height
