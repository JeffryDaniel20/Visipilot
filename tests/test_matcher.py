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


def make_state(elements, image_path="x.png"):
    meta = ScreenshotMeta(viewport_width=1280, viewport_height=800)
    screenshot = Screenshot(image_path=image_path, image_hash="abc", width_px=1280, height_px=800, meta=meta)
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


def _make_screenshot_image(tmp_path, size, regions):
    """Build a white PNG with `regions` (list of (bbox, rgb)) painted on
    top, for real fill-color-based matcher tests.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for bbox, rgb in regions:
        draw.rectangle([bbox.x, bbox.y, bbox.x2, bbox.y2], fill=rgb)
    path = tmp_path / "shot.png"
    img.save(path)
    return str(path)


def test_fill_bonus_disambiguates_bare_click_phrase_real_colors(tmp_path):
    # Reproduces the real, previously-tied CLICK-step case: "Search:"
    # (white input) vs "Search" (blue button), bare phrase with no
    # structural word so _aspect_ratio_bonus does not apply.
    white_input = make_element("inp", "Search:", x=0, y=0, w=350, h=30, etype=ElementType.TEXT_INPUT)
    blue_button = make_element("btn", "Search", x=400, y=0, w=80, h=30, etype=ElementType.BUTTON)
    image_path = _make_screenshot_image(
        tmp_path, (1280, 800), [(blue_button.bbox, (43, 92, 255))]
    )
    state = make_state([white_input, blue_button], image_path=image_path)

    results = match_target("Search", state)

    assert results[0].element.id == "btn"
    assert results[0].score > results[1].score


def test_fill_bonus_does_not_reintroduce_tie_for_structural_phrase(tmp_path):
    # The FIND step's "the search box" phrase must still resolve to the
    # input even with a colorful button present in the same real image —
    # _fill_bonus is gated off when a structural word is present, exactly
    # so it can't fight _aspect_ratio_bonus's already-correct answer.
    white_input = make_element("inp", "Search:", x=0, y=0, w=350, h=30, etype=ElementType.TEXT_INPUT)
    blue_button = make_element("btn", "Search", x=400, y=0, w=80, h=30, etype=ElementType.BUTTON)
    image_path = _make_screenshot_image(
        tmp_path, (1280, 800), [(blue_button.bbox, (43, 92, 255))]
    )
    state = make_state([white_input, blue_button], image_path=image_path)

    results = match_target("the search box", state)

    assert results[0].element.id == "inp"


def test_genuine_tie_between_two_identically_styled_buttons_is_preserved():
    # Critical safety-preservation check: when two candidates really are
    # indistinguishable (same text, same shape, same — untestable-here —
    # fill, since no image is sampled), the matcher must still report an
    # exact tie so resolve_single_candidate() correctly refuses to guess.
    # This must keep passing after the fill-bonus change; it's not a
    # weakening of the no-blind-clicking rule, just a narrower set of
    # cases that remain genuinely ambiguous.
    btn_a = make_element("a", "Search", x=0, y=0, w=80, h=30, etype=ElementType.BUTTON)
    btn_b = make_element("b", "Search", x=500, y=0, w=80, h=30, etype=ElementType.BUTTON)
    state = make_state([btn_a, btn_b])  # default "x.png" -> no image, fill_bonus contributes 0 to both

    results = match_target("Search", state)

    assert len(results) == 2
    assert results[0].score == results[1].score


def test_fill_bonus_gracefully_ignored_when_screenshot_missing():
    # No real image on disk (default "x.png") must not raise or otherwise
    # break matching -- the signal is a refinement, never a requirement.
    btn = make_element("btn", "Search", x=0, y=0, w=80, h=30)
    state = make_state([btn])
    results = match_target("Search", state)
    assert results[0].element.id == "btn"


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


# --- textless-icon-neighbor redirect ----------------------------------------
# Regression coverage for the real "confidently wrong click" investigation on
# search_small_icons.html — see implementation-plan.md Phase B for the full
# root-cause writeup: a 16x16 icon button has no OCR text, so a bare "Search"
# phrase only matched the input beside it, and the system clicked the wrong
# (but real, in-viewport) element with full confidence instead of refusing.

def test_textless_icon_neighbor_ties_with_matched_input_for_bare_phrase():
    matched_input = make_element(
        "input", "Search:", x=0, y=0, w=300, h=30, etype=ElementType.TEXT_INPUT,
        relations=[Relation(kind="nearby", target_id="icon")],
    )
    icon_button = make_element(
        "icon", None, x=310, y=0, w=16, h=16, etype=ElementType.BUTTON,
        relations=[Relation(kind="nearby", target_id="input")],
    )
    state = make_state([matched_input, icon_button])

    results = match_target("Search", state)

    scores = {c.element.id: c.score for c in results}
    assert scores["input"] == scores["icon"], "must tie, not silently prefer one"


def test_textless_icon_neighbor_redirect_gated_off_for_structural_phrase():
    # "the search box" already resolves via _aspect_ratio_bonus; the
    # redirect must not re-introduce a tie there.
    matched_input = make_element(
        "input", "Search:", x=0, y=0, w=300, h=30, etype=ElementType.TEXT_INPUT,
        relations=[Relation(kind="nearby", target_id="icon")],
    )
    icon_button = make_element(
        "icon", None, x=310, y=0, w=16, h=16, etype=ElementType.BUTTON,
        relations=[Relation(kind="nearby", target_id="input")],
    )
    state = make_state([matched_input, icon_button])

    results = match_target("the search box", state)

    assert len(results) == 1
    assert results[0].element.id == "input"


def test_textless_icon_neighbor_redirect_ignores_neighbor_with_text():
    # A nearby interactable element that DOES have its own text is not
    # "invisible" to matching and must not be redirected to.
    matched_input = make_element(
        "input", "Search:", x=0, y=0, w=300, h=30, etype=ElementType.TEXT_INPUT,
        relations=[Relation(kind="nearby", target_id="other_btn")],
    )
    other_btn = make_element(
        "other_btn", "Cancel", x=310, y=0, w=60, h=30,
        relations=[Relation(kind="nearby", target_id="input")],
    )
    state = make_state([matched_input, other_btn])

    results = match_target("Search", state)

    ids = [c.element.id for c in results]
    assert ids == ["input"]  # "Cancel" scores on its own merits (zero here), not via redirect


def test_textless_icon_neighbor_redirect_excludes_overlapping_spurious_box():
    # Regression test for the exact bug this feature had before shipping:
    # a large box that heavily OVERLAPS the matched element (a spurious
    # duplicate detection of roughly the same region, not a distinct
    # adjacent control) must not be picked over a small box that's
    # genuinely beside it with near-zero overlap.
    matched_input = make_element(
        "input", "Search:", x=20, y=20, w=300, h=30, etype=ElementType.TEXT_INPUT,
        relations=[Relation(kind="nearby", target_id="spurious"), Relation(kind="nearby", target_id="real_icon")],
    )
    spurious_overlapping_box = make_element(
        "spurious", None, x=0, y=0, w=400, h=400, etype=ElementType.BUTTON,
        relations=[Relation(kind="nearby", target_id="input")],
    )
    real_icon = make_element(
        "real_icon", None, x=325, y=25, w=16, h=16, etype=ElementType.BUTTON,
        relations=[Relation(kind="nearby", target_id="input")],
    )
    state = make_state([matched_input, spurious_overlapping_box, real_icon])

    results = match_target("Search", state)

    scores = {c.element.id: c.score for c in results}
    assert scores["input"] == scores["real_icon"]
    assert scores.get("spurious", 0) == 0
