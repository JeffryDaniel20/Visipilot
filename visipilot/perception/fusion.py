"""Fusion: merge detector + OCR output into a deduplicated UIElement list.

Pure function, no model dependency — testable with synthetic UIElements
independent of whether the detector/OCR models are even installed.
"""
from __future__ import annotations

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
    """
    used_ocr_ids: set[str] = set()
    fused: list[UIElement] = []

    for det in detector_elements:
        matched_texts: list[str] = []
        for ocr_el in ocr_elements:
            if ocr_el.id in used_ocr_ids:
                continue
            if _bbox_overlap_ratio(det.bbox, ocr_el.bbox) >= overlap_threshold:
                if ocr_el.text:
                    matched_texts.append(ocr_el.text)
                used_ocr_ids.add(ocr_el.id)

        text = " ".join(matched_texts) if matched_texts else det.text
        source = ElementSource.FUSION if matched_texts else det.source
        fused.append(det.model_copy(update={"text": text, "source": source}))

    for ocr_el in ocr_elements:
        if ocr_el.id not in used_ocr_ids:
            fused.append(ocr_el)

    return fused
