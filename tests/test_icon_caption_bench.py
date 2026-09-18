"""Unit tests for the pure, synthetic-input parts of
visipilot.eval.icon_caption_bench — the candidate-narrowing predicate
used to pick which detected regions get captioned. Deliberately does not
exercise Florence-2 itself (network + GPU + currently broken upstream
compatibility, see implementation-plan.md B.7) — this only locks in that
the benchmark reuses B.5's real narrowing criteria correctly, so results
from the (currently blocked) captioning stage would be trustworthy once
it's unblocked.
"""
from __future__ import annotations

from visipilot.eval.icon_caption_bench import _is_icon_candidate
from visipilot.target_selection.matcher import _NEIGHBOR_MAX_DIMENSION_PX
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


def test_compact_textless_interactable_is_a_candidate():
    el = make_element("icon", None, w=16, h=16)
    assert _is_icon_candidate(el) is True


def test_element_with_text_is_not_a_candidate():
    # Has its own OCR text -- not the "invisible to text matching" case
    # this whole investigation is about.
    el = make_element("btn", "Search", w=16, h=16)
    assert _is_icon_candidate(el) is False


def test_non_interactable_is_not_a_candidate():
    el = make_element("deco", None, w=16, h=16, interactable=False)
    assert _is_icon_candidate(el) is False


def test_oversized_element_is_not_a_candidate():
    # Reuses the exact B.5 size cap so the benchmark can't accidentally
    # caption a large spurious region and call it an "icon".
    el = make_element("big", None, w=_NEIGHBOR_MAX_DIMENSION_PX + 1, h=16)
    assert _is_icon_candidate(el) is False


def test_element_at_exact_size_cap_is_a_candidate():
    el = make_element("edge", None, w=_NEIGHBOR_MAX_DIMENSION_PX, h=_NEIGHBOR_MAX_DIMENSION_PX)
    assert _is_icon_candidate(el) is True
