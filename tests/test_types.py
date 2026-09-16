"""Unit tests for visipilot.types — schema validation and serialization
round-trips (TESTING.md #1).
"""
from __future__ import annotations

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


def make_screenshot() -> Screenshot:
    meta = ScreenshotMeta(viewport_width=1280, viewport_height=800, device_scale_factor=1.5)
    return Screenshot(
        image_path="out/screenshots/abc123.png",
        image_hash="abc123",
        width_px=1920,
        height_px=1200,
        meta=meta,
    )


def test_bbox_derived_properties():
    bbox = BBox(x=10, y=20, width=100, height=50, space=CoordinateSpace.SCREENSHOT_PX)
    assert bbox.x2 == 110
    assert bbox.y2 == 70
    assert bbox.center == (60, 45)


def test_screenshot_meta_effective_scale():
    meta = ScreenshotMeta(viewport_width=1280, viewport_height=800, device_scale_factor=1.5, zoom_factor=1.25)
    assert meta.effective_scale == 1.875


def test_ui_element_round_trip():
    element = UIElement(
        id="e1",
        type=ElementType.BUTTON,
        bbox=BBox(x=0, y=0, width=10, height=10, space=CoordinateSpace.SCREENSHOT_PX),
        text="Search",
        confidence=0.95,
        interactable=True,
        semantic_role="search-button",
        relations=[Relation(kind="label_of", target_id="e2")],
        source=ElementSource.DETECTOR,
    )
    dumped = element.model_dump_json()
    restored = UIElement.model_validate_json(dumped)
    assert restored == element


def test_semantic_ui_state_round_trip():
    screenshot = make_screenshot()
    state = SemanticUIState(
        elements=[
            UIElement(
                id="e1",
                type=ElementType.TEXT_INPUT,
                bbox=BBox(x=1, y=2, width=3, height=4, space=CoordinateSpace.SCREENSHOT_PX),
                confidence=0.8,
                interactable=True,
                source=ElementSource.FUSION,
            )
        ],
        screenshot=screenshot,
    )
    dumped = state.model_dump_json()
    restored = SemanticUIState.model_validate_json(dumped)
    assert restored == state


def test_ui_element_defaults():
    element = UIElement(
        id="e1",
        type=ElementType.UNKNOWN,
        bbox=BBox(x=0, y=0, width=1, height=1, space=CoordinateSpace.SCREENSHOT_PX),
        confidence=0.5,
        interactable=False,
        source=ElementSource.OCR,
    )
    assert element.text is None
    assert element.relations == []
    assert element.semantic_role is None
