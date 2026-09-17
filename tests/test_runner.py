"""Unit tests for visipilot.action.runner — mocked Playwright page,
synthetic SemanticUIState.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from visipilot.action.runner import run_steps
from visipilot.target_selection.instruction_parser import parse_instruction
from visipilot.types import (
    ActionOutcome,
    BBox,
    CoordinateSpace,
    ElementSource,
    ElementType,
    Screenshot,
    ScreenshotMeta,
    SemanticUIState,
    UIElement,
)


def make_element(id_, text, x, y, w=80, h=30, etype=ElementType.BUTTON, interactable=True):
    return UIElement(
        id=id_,
        type=etype,
        bbox=BBox(x=x, y=y, width=w, height=h, space=CoordinateSpace.SCREENSHOT_PX),
        text=text,
        confidence=0.9,
        interactable=interactable,
        source=ElementSource.FUSION,
    )


def make_state(elements, dpr=1.0):
    meta = ScreenshotMeta(viewport_width=1280, viewport_height=800, device_scale_factor=dpr)
    screenshot = Screenshot(image_path="x.png", image_hash="abc", width_px=int(1280 * dpr), height_px=int(800 * dpr), meta=meta)
    return SemanticUIState(elements=elements, screenshot=screenshot)


def test_unambiguous_click_only_instruction_succeeds():
    subscribe = make_element("btn", "Subscribe", x=100, y=100)
    state = make_state([subscribe])
    steps = parse_instruction("Click Subscribe.")
    page = MagicMock()

    records = run_steps(page, steps, state)

    assert len(records) == 1
    assert records[0].outcome == ActionOutcome.SUCCESS
    assert records[0].target_element_id == "btn"
    page.mouse.click.assert_called_once()


def test_find_then_type_sequences_correctly():
    search_input = make_element("inp", "Search:", x=0, y=0, w=300, h=30, etype=ElementType.TEXT_INPUT)
    state = make_state([search_input])
    steps = parse_instruction("Find the search box, type Python.")
    page = MagicMock()

    records = run_steps(page, steps, state)

    assert [r.action for r in records] == ["find", "type"]
    assert records[0].outcome == ActionOutcome.SUCCESS
    assert records[1].outcome == ActionOutcome.SUCCESS
    assert records[1].value == "Python"
    page.keyboard.type.assert_called_once_with("Python")


def test_stops_at_ambiguous_click_step_without_clicking():
    tied_a = make_element("a", "Search", x=0, y=0)
    tied_b = make_element("b", "Search", x=500, y=0)
    state = make_state([tied_a, tied_b])
    steps = parse_instruction("Click Search.")
    page = MagicMock()

    records = run_steps(page, steps, state)

    assert len(records) == 1
    assert records[0].outcome == ActionOutcome.FAILED_AMBIGUOUS
    page.mouse.click.assert_not_called()


def test_type_without_prior_find_fails_safely():
    state = make_state([])
    steps = parse_instruction("Type Python.")
    page = MagicMock()

    records = run_steps(page, steps, state)

    assert len(records) == 1
    assert records[0].outcome == ActionOutcome.FAILED_NO_TARGET
    page.keyboard.type.assert_not_called()


def test_full_vertical_slice_instruction_halts_at_unresolved_click():
    # The FIND step resolves (search box is wide, aspect-ratio bonus
    # applies); the CLICK step is a genuine tie between two same-text
    # candidates and must halt rather than guess.
    search_input = make_element("inp", "Search:", x=0, y=0, w=350, h=30, etype=ElementType.TEXT_INPUT)
    tied_btn_a = make_element("btn-a", "Search", x=400, y=0, w=60, h=30)
    tied_btn_b = make_element("btn-b", "Search", x=600, y=0, w=60, h=30)
    state = make_state([search_input, tied_btn_a, tied_btn_b])
    steps = parse_instruction("Find the search box, type Python, and click Search.")
    page = MagicMock()

    records = run_steps(page, steps, state)

    assert [r.action for r in records] == ["find", "type", "click"]
    assert records[0].outcome == ActionOutcome.SUCCESS
    assert records[1].outcome == ActionOutcome.SUCCESS
    assert records[2].outcome == ActionOutcome.FAILED_AMBIGUOUS
    page.mouse.click.assert_called_once()  # only the "find"-then-"type" focus click, never a click on "Search"
