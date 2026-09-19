"""Unit tests for visipilot.action.executor — mocked Playwright page, no
browser required.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from visipilot.action.executor import (
    AmbiguousTargetError,
    NoConfidentTargetError,
    OrdinalOutOfRangeError,
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


# --- resolve_single_candidate(ordinal=...) — "the second result" ----------

def test_ordinal_picks_by_reading_order_not_score_order():
    # Deliberately out of both id-alphabetical and score-descending order
    # in the input list -- only y-then-x position should determine which
    # is "first"/"second"/"third".
    top = make_candidate("bottom", 0.9, y=300)
    mid = make_candidate("top", 0.9, y=100)
    bot = make_candidate("middle", 0.9, y=200)
    candidates = [top, mid, bot]

    assert resolve_single_candidate(candidates, ordinal=1).element.id == "top"
    assert resolve_single_candidate(candidates, ordinal=2).element.id == "middle"
    assert resolve_single_candidate(candidates, ordinal=3).element.id == "bottom"


def test_ordinal_breaks_ties_left_to_right_on_the_same_row():
    left = make_candidate("left", 0.9, x=0, y=100)
    right = make_candidate("right", 0.9, x=300, y=100)
    candidates = [right, left]

    assert resolve_single_candidate(candidates, ordinal=1).element.id == "left"
    assert resolve_single_candidate(candidates, ordinal=2).element.id == "right"


def test_ordinal_last_is_negative_one():
    candidates = [
        make_candidate("first", 0.9, y=100),
        make_candidate("second", 0.9, y=200),
        make_candidate("third", 0.9, y=300),
    ]
    assert resolve_single_candidate(candidates, ordinal=-1).element.id == "third"


def test_ordinal_out_of_range_raises_instead_of_guessing():
    # Real "missing ordinal position" case: asking for the third of only
    # two tied candidates must refuse, never fall back to the closest one.
    candidates = [make_candidate("a", 0.9, y=100), make_candidate("b", 0.9, y=200)]
    with pytest.raises(OrdinalOutOfRangeError) as exc_info:
        resolve_single_candidate(candidates, ordinal=3)
    assert exc_info.value.ordinal == 3
    assert exc_info.value.available == 2


def test_ordinal_out_of_range_on_a_single_unambiguous_candidate():
    # "the second X" when there's only ever been one X at all.
    candidates = [make_candidate("only", 0.9)]
    with pytest.raises(OrdinalOutOfRangeError) as exc_info:
        resolve_single_candidate(candidates, ordinal=2)
    assert exc_info.value.available == 1


def test_ordinal_one_on_a_single_unambiguous_candidate_still_resolves():
    # ordinal=1 must not require an actual tie -- "the first Subscribe
    # button" when there's trivially only one is not an error.
    candidates = [make_candidate("only", 0.9)]
    result = resolve_single_candidate(candidates, ordinal=1)
    assert result.element.id == "only"


def test_ordinal_never_reaches_past_the_top_scoring_tied_group():
    # A lower-scored, less-relevant candidate must never be treated as
    # the ordinal's "third" option just because it exists in the list --
    # ordinal only ever indexes within the group actually tied for top.
    tied_a = make_candidate("tied-a", 0.9, y=100)
    tied_b = make_candidate("tied-b", 0.9, y=200)
    unrelated = make_candidate("unrelated", 0.4, y=300)
    candidates = [tied_a, tied_b, unrelated]

    with pytest.raises(OrdinalOutOfRangeError) as exc_info:
        resolve_single_candidate(candidates, ordinal=3, tie_epsilon=0.05)
    assert exc_info.value.available == 2


def test_ordinal_below_min_confidence_still_refuses_via_no_confident_target():
    candidates = [make_candidate("a", 0.1)]
    with pytest.raises(NoConfidentTargetError):
        resolve_single_candidate(candidates, min_confidence=0.3, ordinal=1)


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
