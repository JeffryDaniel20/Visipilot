"""Unit tests for visipilot.action.executor — mocked Playwright page, no
browser required.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from visipilot.action.executor import (
    AmbiguousTargetError,
    NoConfidentTargetError,
    click_element,
    resolve_single_candidate,
    type_into_element,
)
from visipilot.target_selection.matcher import MatchCandidate
from visipilot.types import ActionOutcome, BBox, CoordinateSpace, ElementSource, ElementType, ScreenshotMeta, UIElement


def make_candidate(id_, score, x=100, y=100, w=50, h=20):
    element = UIElement(
        id=id_,
        type=ElementType.BUTTON,
        bbox=BBox(x=x, y=y, width=w, height=h, space=CoordinateSpace.SCREENSHOT_PX),
        confidence=0.9,
        interactable=True,
        source=ElementSource.FUSION,
    )
    return MatchCandidate(element=element, score=score)


def make_meta(dpr=1.0, viewport_width=1280, viewport_height=800):
    return ScreenshotMeta(viewport_width=viewport_width, viewport_height=viewport_height, device_scale_factor=dpr)


# --- resolve_single_candidate ---------------------------------------------

def test_resolve_single_candidate_clear_winner():
    candidates = [make_candidate("a", 0.9), make_candidate("b", 0.3)]
    result = resolve_single_candidate(candidates)
    assert result.element.id == "a"


def test_resolve_single_candidate_no_candidates_raises():
    with pytest.raises(NoConfidentTargetError):
        resolve_single_candidate([])


def test_resolve_single_candidate_below_min_confidence_raises():
    candidates = [make_candidate("a", 0.1)]
    with pytest.raises(NoConfidentTargetError):
        resolve_single_candidate(candidates, min_confidence=0.3)


def test_resolve_single_candidate_tied_raises_ambiguous():
    candidates = [make_candidate("a", 0.6), make_candidate("b", 0.58)]
    with pytest.raises(AmbiguousTargetError) as exc_info:
        resolve_single_candidate(candidates, tie_epsilon=0.05)
    assert len(exc_info.value.tied_candidates) == 2


def test_resolve_single_candidate_not_tied_when_gap_exceeds_epsilon():
    candidates = [make_candidate("a", 0.9), make_candidate("b", 0.5)]
    result = resolve_single_candidate(candidates, tie_epsilon=0.05)
    assert result.element.id == "a"


# --- click_element ----------------------------------------------------------

def test_click_element_calls_trusted_mouse_click_at_grounded_point():
    page = MagicMock()
    candidate = make_candidate("btn", 0.9, x=100, y=200, w=40, h=20)
    meta = make_meta(dpr=2.0)

    record = click_element(page, candidate, meta)

    # bbox center in screenshot px = (120, 210); at DPR=2 -> viewport css (60, 105)
    page.mouse.click.assert_called_once_with(60.0, 105.0)
    assert record.outcome == ActionOutcome.SUCCESS
    assert record.click_point == (60.0, 105.0)
    assert record.target_element_id == "btn"


def test_click_element_out_of_viewport_does_not_click():
    page = MagicMock()
    candidate = make_candidate("btn", 0.9, x=10000, y=10000, w=10, h=10)
    meta = make_meta()

    record = click_element(page, candidate, meta)

    page.mouse.click.assert_not_called()
    assert record.outcome == ActionOutcome.FAILED_OUT_OF_VIEWPORT


def test_click_element_playwright_exception_captured():
    page = MagicMock()
    page.mouse.click.side_effect = RuntimeError("boom")
    candidate = make_candidate("btn", 0.9)
    meta = make_meta()

    record = click_element(page, candidate, meta)

    assert record.outcome == ActionOutcome.FAILED_EXECUTION_ERROR
    assert "boom" in record.error_message


# --- type_into_element -------------------------------------------------------

def test_type_into_element_clicks_then_types():
    page = MagicMock()
    candidate = make_candidate("inp", 0.9, x=0, y=0, w=100, h=20)
    meta = make_meta()

    record = type_into_element(page, candidate, meta, "Python")

    page.mouse.click.assert_called_once()
    page.keyboard.type.assert_called_once_with("Python")
    assert record.outcome == ActionOutcome.SUCCESS
    assert record.value == "Python"


def test_type_into_element_out_of_viewport_skips_typing():
    page = MagicMock()
    candidate = make_candidate("inp", 0.9, x=10000, y=10000, w=10, h=10)
    meta = make_meta()

    record = type_into_element(page, candidate, meta, "Python")

    page.keyboard.type.assert_not_called()
    assert record.outcome == ActionOutcome.FAILED_OUT_OF_VIEWPORT
