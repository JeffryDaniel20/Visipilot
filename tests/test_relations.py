"""Unit tests for visipilot.semantic.relations — pure logic, synthetic
layouts.
"""
from __future__ import annotations

from visipilot.semantic.relations import infer_relations
from visipilot.types import BBox, CoordinateSpace, ElementSource, ElementType, UIElement


def make_element(id_, x, y, w, h, etype=ElementType.BUTTON, interactable=True, source=ElementSource.DETECTOR):
    return UIElement(
        id=id_,
        type=etype,
        bbox=BBox(x=x, y=y, width=w, height=h, space=CoordinateSpace.SCREENSHOT_PX),
        confidence=0.9,
        interactable=interactable,
        source=source,
    )


def relation_kinds(element, target_id=None):
    if target_id is None:
        return {r.kind for r in element.relations}
    return {r.kind for r in element.relations if r.target_id == target_id}


def test_contained_in_detected():
    outer = make_element("outer", 0, 0, 200, 200)
    inner = make_element("inner", 50, 50, 20, 20)
    result = infer_relations([outer, inner])
    inner_out, outer_out = result[1], result[0]
    assert "contained_in" in relation_kinds(inner_out, "outer")
    assert "contained_in" not in relation_kinds(outer_out, "inner")


def test_label_of_for_nonoverlapping_nearby_text():
    label = make_element("label", 0, 0, 40, 20, etype=ElementType.TEXT, interactable=False, source=ElementSource.OCR)
    button = make_element("btn", 50, 0, 60, 20, etype=ElementType.BUTTON, interactable=True)
    result = infer_relations([label, button], nearby_distance_px=60.0)
    label_out = result[0]
    assert "label_of" in relation_kinds(label_out, "btn")


def test_label_of_not_created_when_overlapping():
    # text overlapping the button's own bbox should NOT get label_of —
    # fusion is expected to have already absorbed overlapping text.
    label = make_element("label", 10, 5, 20, 10, etype=ElementType.TEXT, interactable=False, source=ElementSource.OCR)
    button = make_element("btn", 0, 0, 60, 20, etype=ElementType.BUTTON, interactable=True)
    result = infer_relations([label, button])
    label_out = result[0]
    assert "label_of" not in relation_kinds(label_out, "btn")


def test_label_of_not_created_when_too_far():
    label = make_element("label", 0, 0, 40, 20, etype=ElementType.TEXT, interactable=False, source=ElementSource.OCR)
    button = make_element("btn", 500, 500, 60, 20, etype=ElementType.BUTTON, interactable=True)
    result = infer_relations([label, button], nearby_distance_px=60.0)
    label_out = result[0]
    assert "label_of" not in relation_kinds(label_out, "btn")
    assert "nearby" not in relation_kinds(label_out, "btn")


def test_nearby_symmetric_within_threshold():
    a = make_element("a", 0, 0, 20, 20)
    b = make_element("b", 40, 0, 20, 20)  # gap = 20px
    result = infer_relations([a, b], nearby_distance_px=30.0)
    a_out, b_out = result
    assert "nearby" in relation_kinds(a_out, "b")
    assert "nearby" in relation_kinds(b_out, "a")


def test_nearby_not_added_beyond_threshold():
    a = make_element("a", 0, 0, 20, 20)
    b = make_element("b", 1000, 1000, 20, 20)
    result = infer_relations([a, b], nearby_distance_px=30.0)
    a_out, b_out = result
    assert relation_kinds(a_out, "b") == set()
    assert relation_kinds(b_out, "a") == set()


def test_no_duplicate_relation_when_both_specific_and_nearby_would_apply():
    outer = make_element("outer", 0, 0, 200, 200)
    inner = make_element("inner", 50, 50, 20, 20)
    result = infer_relations([outer, inner], nearby_distance_px=1000.0)
    inner_out, outer_out = result[1], result[0]
    # inner is contained in outer; must not ALSO get a "nearby" to outer.
    assert relation_kinds(inner_out, "outer") == {"contained_in"}
