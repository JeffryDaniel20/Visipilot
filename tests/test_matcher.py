"""Unit tests for visipilot.target_selection.matcher — synthetic
SemanticUIState fixtures.
"""
from __future__ import annotations

from visipilot.target_selection.matcher import match_target
from visipilot.types import (
    BBox,
    CoordinateSpace,
    ElementSource,
    ElementType,
    Relation,
    Screenshot,
    ScreenshotMeta,
    SemanticUIState,
    UIElement,
)


def make_element(id_, text, x=0, y=0, w=50, h=20, etype=ElementType.BUTTON, interactable=True, relations=None):
    return UIElement(
        id=id_,
        type=etype,
        bbox=BBox(x=x, y=y, width=w, height=h, space=CoordinateSpace.SCREENSHOT_PX),
        text=text,
        confidence=0.9,
        interactable=interactable,
        relations=relations or [],
        source=ElementSource.FUSION,
    )


def make_state(elements):
    meta = ScreenshotMeta(viewport_width=1280, viewport_height=800)
    screenshot = Screenshot(image_path="x.png", image_hash="abc", width_px=1280, height_px=800, meta=meta)
    return SemanticUIState(elements=elements, screenshot=screenshot)


def test_exact_text_match_wins():
    search_btn = make_element("btn", "Search", x=100)
    subscribe_btn = make_element("sub", "Subscribe", x=200)
    state = make_state([search_btn, subscribe_btn])

    results = match_target("Search", state)

    assert results[0].element.id == "btn"
    assert results[0].score > results[1].score if len(results) > 1 else True


def test_structural_word_prefers_multiword_over_single_word_button():
    input_el = make_element("inp", "Search:", etype=ElementType.TEXT_INPUT, x=0)
    button_el = make_element("btn", "Search", etype=ElementType.BUTTON, x=200)
    state = make_state([input_el, button_el])

    results = match_target("the search box", state)

    ids_in_order = [c.element.id for c in results]
    assert ids_in_order[0] == "inp"


def test_aspect_ratio_disambiguates_wide_input_from_compact_button():
    # Regression coverage for the real ambiguity found in the previous
    # milestone: "Search:" (input placeholder) and "Search" (button
    # label) tokenize identically, so _structural_bonus alone (which
    # only looks at word count) can't tell them apart. Real geometry —
    # the input is much wider relative to its height — should.
    wide_input = make_element("inp", "Search:", etype=ElementType.TEXT_INPUT, x=0, w=355, h=41)
    compact_button = make_element("btn", "Search", etype=ElementType.BUTTON, x=400, w=87, h=39)
    state = make_state([wide_input, compact_button])

    results = match_target("the search box", state)

    assert results[0].element.id == "inp"
    assert results[0].score > results[1].score


def test_no_match_returns_empty():
    el = make_element("btn", "Subscribe")
    state = make_state([el])
    results = match_target("nonexistent phrase xyz", state)
    assert results == []


def test_top_k_respected():
    elements = [make_element(f"e{i}", "search", x=i * 10) for i in range(5)]
    state = make_state(elements)
    results = match_target("search", state, top_k=2)
    assert len(results) == 2


def test_long_paragraph_incidentally_containing_phrase_words_loses_to_short_real_match():
    # Regression test for a real bug found running the full pipeline
    # against the controlled test page: a long distractor paragraph that
    # happens to contain the target phrase's words ranked above the
    # actual short-text control until the length penalty was added.
    distractor = make_element(
        "distractor",
        "This page also contains distractor elements that pixels-first "
        "perception pipeline must correctly ignore when asked to find the search box",
        etype=ElementType.TEXT,
        interactable=False,
    )
    real_input = make_element("input", "Search:", etype=ElementType.TEXT_INPUT)
    state = make_state([distractor, real_input])

    results = match_target("search box", state)

    assert results[0].element.id == "input"


def test_label_of_relation_redirects_score_to_target():
    label = make_element(
        "label", "Search:", etype=ElementType.TEXT, interactable=False,
        relations=[Relation(kind="label_of", target_id="input")],
    )
    input_el = make_element("input", None, etype=ElementType.TEXT_INPUT)
    state = make_state([label, input_el])

    results = match_target("search", state)

    ids = [c.element.id for c in results]
    assert "input" in ids
