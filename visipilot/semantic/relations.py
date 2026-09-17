"""Relation inference: nearby / label_of / contained_in between UIElements.

Pure function, no model dependency — operates purely on already-computed
bboxes (post-fusion), all in SCREENSHOT_PX space.
"""
from __future__ import annotations

from visipilot.types import BBox, ElementType, Relation, UIElement

DEFAULT_NEARBY_DISTANCE_PX = 60.0


def _contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer.x <= inner.x
        and outer.y <= inner.y
        and outer.x2 >= inner.x2
        and outer.y2 >= inner.y2
        and (outer.width * outer.height) > (inner.width * inner.height)
    )


def _edge_gap(a: BBox, b: BBox) -> float:
    """Shortest gap between the two boxes' edges. 0 (or less, clamped to
    0) if they overlap.
    """
    dx = max(a.x - b.x2, b.x - a.x2, 0.0)
    dy = max(a.y - b.y2, b.y - a.y2, 0.0)
    return (dx**2 + dy**2) ** 0.5


def infer_relations(
    elements: list[UIElement],
    nearby_distance_px: float = DEFAULT_NEARBY_DISTANCE_PX,
) -> list[UIElement]:
    """Return a new list of elements with `relations` populated.

    - contained_in: element A's bbox is strictly inside a larger element
      B's bbox -> A gets a `contained_in` relation to B.
    - label_of: a non-interactable TEXT element whose bbox does NOT
      overlap an interactable element, but sits within
      `nearby_distance_px` of one -> the text element gets a `label_of`
      relation to that interactable element (the "caption next to a
      control" pattern). On pages where fusion already absorbed the OCR
      text directly into the control's own bbox, no standalone TEXT
      element remains for this pattern to fire on — see
      implementation-plan.md for what was actually observed on the
      controlled test page.
    - nearby: any two elements whose edge-to-edge gap is within
      `nearby_distance_px` get a symmetric `nearby` relation, unless a
      more specific relation (contained_in/label_of) already covers that
      pair.
    """
    n = len(elements)
    relations: list[list[Relation]] = [[] for _ in range(n)]
    specific_pairs: set[tuple[int, int]] = set()

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _contains(elements[j].bbox, elements[i].bbox):
                relations[i].append(Relation(kind="contained_in", target_id=elements[j].id))
                specific_pairs.add((i, j))
                specific_pairs.add((j, i))

    for i in range(n):
        a = elements[i]
        if a.interactable or a.type != ElementType.TEXT:
            continue
        for j in range(n):
            if i == j or not elements[j].interactable:
                continue
            b = elements[j]
            gap = _edge_gap(a.bbox, b.bbox)
            if 0 < gap <= nearby_distance_px and (i, j) not in specific_pairs:
                relations[i].append(Relation(kind="label_of", target_id=b.id))
                specific_pairs.add((i, j))
                specific_pairs.add((j, i))

    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in specific_pairs:
                continue
            if _edge_gap(elements[i].bbox, elements[j].bbox) <= nearby_distance_px:
                relations[i].append(Relation(kind="nearby", target_id=elements[j].id))
                relations[j].append(Relation(kind="nearby", target_id=elements[i].id))

    return [el.model_copy(update={"relations": relations[i]}) for i, el in enumerate(elements)]
