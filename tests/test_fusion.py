"""Unit tests for visipilot.perception.fusion — pure logic, no model
dependency, so these run regardless of whether the detector/OCR models
are installed.
"""
from __future__ import annotations

from visipilot.perception.fusion import fuse
from visipilot.types import BBox, CoordinateSpace, ElementSource, ElementType, UIElement


def make_element(id_, x, y, w, h, source, text=None, etype=ElementType.BUTTON):
    return UIElement(
        id=id_,
        type=etype,
        bbox=BBox(x=x, y=y, width=w, height=h, space=CoordinateSpace.SCREENSHOT_PX),
        text=text,
        confidence=0.9,
        interactable=source == ElementSource.DETECTOR,
        source=source,
    )


def test_fuse_absorbs_overlapping_ocr_text():
    det = make_element("det-0", 0, 0, 100, 40, ElementSource.DETECTOR, etype=ElementType.BUTTON)
    ocr = make_element("ocr-0", 10, 10, 40, 20, ElementSource.OCR, text="Search")

    fused = fuse([det], [ocr])

    assert len(fused) == 1
    assert fused[0].id == "det-0"
    assert fused[0].text == "Search"
    assert fused[0].source == ElementSource.FUSION


def test_fuse_keeps_non_overlapping_ocr_standalone():
    det = make_element("det-0", 0, 0, 100, 40, ElementSource.DETECTOR)
    ocr = make_element("ocr-0", 500, 500, 40, 20, ElementSource.OCR, text="unrelated")

    fused = fuse([det], [ocr])

    assert len(fused) == 2
    ids = {e.id for e in fused}
    assert ids == {"det-0", "ocr-0"}
    standalone = next(e for e in fused if e.id == "ocr-0")
    assert standalone.source == ElementSource.OCR
    assert standalone.text == "unrelated"


def test_fuse_multiple_ocr_boxes_concatenated():
    det = make_element("det-0", 0, 0, 200, 40, ElementSource.DETECTOR)
    ocr1 = make_element("ocr-0", 5, 5, 40, 20, ElementSource.OCR, text="Hello")
    ocr2 = make_element("ocr-1", 60, 5, 40, 20, ElementSource.OCR, text="World")

    fused = fuse([det], [ocr1, ocr2])

    assert len(fused) == 1
    assert fused[0].text == "Hello World"


def test_fuse_no_ocr_keeps_detector_text_none():
    det = make_element("det-0", 0, 0, 100, 40, ElementSource.DETECTOR)
    fused = fuse([det], [])
    assert len(fused) == 1
    assert fused[0].text is None
    assert fused[0].source == ElementSource.DETECTOR


def test_fuse_no_detectors_keeps_all_ocr():
    ocr1 = make_element("ocr-0", 0, 0, 40, 20, ElementSource.OCR, text="a")
    ocr2 = make_element("ocr-1", 100, 100, 40, 20, ElementSource.OCR, text="b")
    fused = fuse([], [ocr1, ocr2])
    assert {e.id for e in fused} == {"ocr-0", "ocr-1"}


def test_fuse_ocr_box_only_partially_overlaps_below_threshold():
    det = make_element("det-0", 0, 0, 20, 20, ElementSource.DETECTOR)
    # ocr box mostly outside det (only 25% overlap by area)
    ocr = make_element("ocr-0", 10, 10, 20, 20, ElementSource.OCR, text="edge")
    fused = fuse([det], [ocr], overlap_threshold=0.5)
    assert len(fused) == 2  # not absorbed, overlap ratio is 0.25 < 0.5
