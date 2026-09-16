"""DOM-derived ground truth extraction — evaluation-only.

Used to score pixels-first runs (did the detected/grounded element
actually match the real DOM element?) and, later, as the DOM/AX baseline
for comparison. Never imported by runtime perception/grounding code.
"""
from __future__ import annotations

from playwright.sync_api import Page

from visipilot.types import BBox, CoordinateSpace


class GroundTruthElementNotFound(RuntimeError):
    pass


def get_element_bbox(page: Page, selector: str) -> BBox:
    """Return the true bounding box of a DOM element, in VIEWPORT_CSS
    space (Playwright's ``bounding_box()`` is already viewport-relative
    and scroll-adjusted, matching CoordinateSpace.VIEWPORT_CSS).
    """
    handle = page.query_selector(selector)
    if handle is None:
        raise GroundTruthElementNotFound(f"no element matches selector {selector!r}")
    box = handle.bounding_box()
    if box is None:
        raise GroundTruthElementNotFound(
            f"element {selector!r} matched but is not visible (no bounding box)"
        )
    return BBox(
        x=box["x"],
        y=box["y"],
        width=box["width"],
        height=box["height"],
        space=CoordinateSpace.VIEWPORT_CSS,
    )


def get_ground_truth(page: Page, selectors: dict[str, str]) -> dict[str, BBox]:
    """Resolve a name -> CSS selector mapping to name -> true BBox.

    Raises GroundTruthElementNotFound for any selector that doesn't
    resolve, rather than silently skipping it — a missing ground-truth
    element usually means the test page changed and the eval harness is
    now scoring against a stale assumption.
    """
    return {name: get_element_bbox(page, sel) for name, sel in selectors.items()}
