"""Fusion: merge detector + OCR output into a deduplicated UIElement list.

Pure function, no model dependency — testable with synthetic UIElements
independent of whether the detector/OCR models are even installed.
"""
from __future__ import annotations

from collections import defaultdict

from visipilot.types import BBox, ElementSource, UIElement


def _bbox_overlap_ratio(a: BBox, b: BBox) -> float:
    """Fraction of b's area that overlaps with a (intersection / b.area)."""
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    b_area = b.width * b.height
    if b_area <= 0:
        return 0.0
    return intersection / b_area


def fuse(
    detector_elements: list[UIElement],
    ocr_elements: list[UIElement],
    overlap_threshold: float = 0.5,
) -> list[UIElement]:
    """Associate OCR text boxes with overlapping detector boxes.

    A detector element absorbs any OCR text element whose bbox overlaps it
    by at least ``overlap_threshold`` (fraction of the OCR box's own area
    that lies inside the detector box); absorbed OCR elements are dropped
    from the standalone output and their text is concatenated onto the
    detector element (whose ``source`` becomes FUSION). OCR elements that
    don't overlap any detector box are kept standalone (source stays OCR)
    — this preserves distractor text (e.g. an unrelated paragraph) that a
    detector correctly did not flag as interactive.

    When an OCR box overlaps *multiple* detector boxes above threshold,
    it is assigned to the one with the **highest** overlap ratio, not
    simply the first one encountered in `detector_elements` order; ties
    (including the common case of two boxes both *fully* containing the
    OCR box, ratio 1.0 each) are broken in favor of the **smaller-area**
    detector box. Real motivating case, found via Phase B testing on a
    small-icon-button page (see implementation-plan.md): the detector
    produced both a precise box tightly matching the real input and a
    spurious, oversized box spanning an unrelated toolbar plus the
    search row; both *fully contained* the input's OCR text (ratio 1.0
    for both — overlap ratio alone doesn't distinguish "tightly fits"
    from "loosely contains"), and first-match ordering let the spurious
    box win purely because it happened to come first in
    `detector_elements`. This changes nothing for the common case where
    only one detector box overlaps a given OCR box.
    """
    best_match: dict[str, tuple[str, float, float]] = {}  # ocr_id -> (det_id, overlap_ratio, det_area)
    for det in detector_elements:
        det_area = det.bbox.width * det.bbox.height
        for ocr_el in ocr_elements:
            ratio = _bbox_overlap_ratio(det.bbox, ocr_el.bbox)
            if ratio < overlap_threshold:
                continue
            current = best_match.get(ocr_el.id)
            if current is None or ratio > current[1] or (ratio == current[1] and det_area < current[2]):
                best_match[ocr_el.id] = (det.id, ratio, det_area)

    assigned_texts: dict[str, list[str]] = defaultdict(list)
    used_ocr_ids: set[str] = set()
    for ocr_el in ocr_elements:
        match = best_match.get(ocr_el.id)
        if match is None:
            continue
        det_id, _ratio, _area = match
        if ocr_el.text:
            assigned_texts[det_id].append(ocr_el.text)
        used_ocr_ids.add(ocr_el.id)

    fused: list[UIElement] = []
    for det in detector_elements:
        matched_texts = assigned_texts.get(det.id, [])
        text = " ".join(matched_texts) if matched_texts else det.text
        source = ElementSource.FUSION if matched_texts else det.source
        fused.append(det.model_copy(update={"text": text, "source": source}))

    for ocr_el in ocr_elements:
        if ocr_el.id not in used_ocr_ids:
            fused.append(ocr_el)

    return fused
