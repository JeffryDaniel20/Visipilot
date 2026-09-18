"""Unit tests for the pure, synthetic-input parts of
visipilot.eval.icon_recall_experiment, plus a regression lock on the
production UIDetector defaults this investigation deliberately did NOT
change — see implementation-plan.md B.8. No network/GPU dependency.
"""
from __future__ import annotations

from visipilot.eval.icon_recall_experiment import _is_compact_candidate, _overlap_ratio
from visipilot.perception.detector import DEFAULT_QUERIES, UIDetector
from visipilot.types import BBox, CoordinateSpace, ElementSource, ElementType, UIElement


def make_element(id_, text, w=16, h=16, interactable=True):
    return UIElement(
        id=id_,
        type=ElementType.BUTTON,
        bbox=BBox(x=0, y=0, width=w, height=h, space=CoordinateSpace.SCREENSHOT_PX),
        text=text,
        confidence=0.9,
        interactable=interactable,
        source=ElementSource.FUSION,
    )


def make_bbox(x, y, w, h):
    return BBox(x=x, y=y, width=w, height=h, space=CoordinateSpace.SCREENSHOT_PX)


def test_overlap_ratio_full_containment_is_one():
    truth = make_bbox(10, 10, 10, 10)
    bbox = make_bbox(0, 0, 40, 40)  # fully contains truth
    assert _overlap_ratio(bbox, truth) == 1.0


def test_overlap_ratio_no_overlap_is_zero():
    truth = make_bbox(0, 0, 10, 10)
    bbox = make_bbox(100, 100, 10, 10)
    assert _overlap_ratio(bbox, truth) == 0.0


def test_overlap_ratio_partial_overlap_is_fraction_of_truth_area():
    truth = make_bbox(0, 0, 10, 10)  # area 100
    bbox = make_bbox(5, 0, 10, 10)   # overlaps x in [5,10) -> 5x10=50
    assert _overlap_ratio(bbox, truth) == 0.5


def test_compact_candidate_predicate_matches_matcher_module():
    # This benchmark's candidate selection must stay identical to B.5's
    # real redirect gating -- otherwise recall/FP numbers measured here
    # wouldn't describe what the runtime pipeline would actually see.
    el = make_element("icon", None, w=16, h=16)
    assert _is_compact_candidate(el) is True
    el_with_text = make_element("btn", "Search", w=16, h=16)
    assert _is_compact_candidate(el_with_text) is False


def test_production_detector_defaults_unchanged_by_this_investigation():
    # Regression lock for implementation-plan.md B.8: the investigation
    # found that both an expanded icon-query vocabulary and a lower
    # confidence threshold raise false-positive compact-candidate counts
    # on pages with no real icon target (measured: 0 -> 9 and 0 -> 19-27
    # respectively across the 5 non-icon Phase B pages), so neither was
    # adopted. If this test ever needs updating, that's a deliberate,
    # evidence-weighed decision to revisit -- not an accidental default
    # change slipped in while chasing one page's recall.
    detector = UIDetector.__new__(UIDetector)  # avoid loading the real model
    assert DEFAULT_QUERIES == {
        "a rectangular text input box": ElementType.TEXT_INPUT,
        "a button": ElementType.BUTTON,
    }
    import inspect
    sig = inspect.signature(UIDetector.__init__)
    assert sig.parameters["score_threshold"].default == 0.1
