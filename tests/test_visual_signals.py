"""Unit tests for visipilot.target_selection.visual_signals — synthetic
images with known fill colors, no browser/model needed.
"""
from __future__ import annotations

from PIL import Image

from visipilot.target_selection.visual_signals import fill_distance_from_white
from visipilot.types import BBox, CoordinateSpace


def make_image(path, size, fill_rgb):
    img = Image.new("RGB", size, fill_rgb)
    img.save(path)
    return str(path)


def full_bbox(w, h):
    return BBox(x=0, y=0, width=w, height=h, space=CoordinateSpace.SCREENSHOT_PX)


def test_white_fill_distance_near_zero(tmp_path):
    path = make_image(tmp_path / "white.png", (100, 40), (255, 255, 255))
    distance = fill_distance_from_white(path, full_bbox(100, 40))
    assert distance < 5.0


def test_saturated_blue_fill_large_distance(tmp_path):
    path = make_image(tmp_path / "blue.png", (100, 40), (43, 92, 255))  # #2b5cff, this project's real button color
    distance = fill_distance_from_white(path, full_bbox(100, 40))
    assert distance > 200.0


def test_neutral_gray_fill_moderate_distance(tmp_path):
    path = make_image(tmp_path / "gray.png", (100, 40), (224, 224, 224))  # #e0e0e0, this project's real secondary button color
    distance = fill_distance_from_white(path, full_bbox(100, 40))
    assert 50.0 < distance < 200.0


def test_near_white_fill_small_distance(tmp_path):
    path = make_image(tmp_path / "nearwhite.png", (100, 40), (251, 251, 251))
    distance = fill_distance_from_white(path, full_bbox(100, 40))
    assert distance < 10.0


def test_zero_area_bbox_returns_zero(tmp_path):
    path = make_image(tmp_path / "white.png", (100, 40), (255, 255, 255))
    zero_bbox = BBox(x=0, y=0, width=0, height=0, space=CoordinateSpace.SCREENSHOT_PX)
    assert fill_distance_from_white(path, zero_bbox) == 0.0
