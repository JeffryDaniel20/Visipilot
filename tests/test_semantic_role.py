"""Unit tests for visipilot.semantic.semantic_role."""
from __future__ import annotations

from visipilot.semantic.semantic_role import infer_semantic_role
from visipilot.types import BBox, CoordinateSpace, ElementSource, ElementType, UIElement


def make_element(text, etype=ElementType.BUTTON, interactable=True):
    return UIElement(
        id="e1",
        type=etype,
        bbox=BBox(x=0, y=0, width=10, height=10, space=CoordinateSpace.SCREENSHOT_PX),
        text=text,
        confidence=0.9,
        interactable=interactable,
        source=ElementSource.FUSION,
    )


def test_search_button():
    el = make_element("Search", etype=ElementType.BUTTON)
    assert infer_semantic_role(el) == "search-button"


def test_search_input():
    el = make_element("Search:", etype=ElementType.TEXT_INPUT)
    assert infer_semantic_role(el) == "search-input"


def test_search_related_other_type():
    el = make_element("Search results", etype=ElementType.TEXT, interactable=False)
    assert infer_semantic_role(el) == "search-related"


def test_submit_button():
    el = make_element("Subscribe", etype=ElementType.BUTTON, interactable=True)
    assert infer_semantic_role(el) == "submit-button"


def test_submit_word_but_not_interactable_is_not_submit_button():
    el = make_element("Subscribe", etype=ElementType.TEXT, interactable=False)
    assert infer_semantic_role(el) != "submit-button"


def test_labeled_input():
    el = make_element("Enter email", etype=ElementType.TEXT_INPUT)
    assert infer_semantic_role(el) == "labeled-input"


def test_no_text_returns_none():
    el = make_element(None)
    assert infer_semantic_role(el) is None


def test_unrelated_text_returns_none():
    el = make_element("This page also contains distractor elements")
    assert infer_semantic_role(el) is None
