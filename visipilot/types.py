"""Typed data structures shared across pipeline stages.

Every inter-stage boundary in VisiPilot passes one of these types, never a
raw dict — see Instructions.md #4 (coding standards) and the architecture
diagram in README.md.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CoordinateSpace(str, Enum):
    """Which pixel/coordinate grid a point or bbox is expressed in.

    - SCREENSHOT_PX: raw pixel coordinates in the captured image file, as
      produced by the detector/OCR stages. Origin is the top-left of the
      screenshot as captured (which may be viewport-clipped or full-page).
    - VIEWPORT_CSS: CSS pixels relative to the current viewport's top-left
      corner. This is what Playwright's mouse/click APIs expect.
    - PAGE_CSS: CSS pixels relative to the full page's top-left corner
      (i.e. VIEWPORT_CSS + current scroll offset).
    """

    SCREENSHOT_PX = "screenshot_px"
    VIEWPORT_CSS = "viewport_css"
    PAGE_CSS = "page_css"


class BBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    space: CoordinateSpace

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


class ScreenshotMeta(BaseModel):
    """Coordinate-space metadata captured alongside a screenshot.

    zoom_factor models browser page zoom as an additional multiplicative
    scale factor on top of device_scale_factor (DPR). This is a modeling
    assumption, not an empirically-confirmed fact about every possible
    browser zoom implementation — it is validated against real Playwright
    screenshots in Phase B's zoom/DPR sweep (see implementation-plan.md);
    revise this model there if reality diverges.
    """

    viewport_width: int
    viewport_height: int
    device_scale_factor: float = 1.0
    zoom_factor: float = 1.0
    scroll_x: float = 0.0
    scroll_y: float = 0.0
    full_page: bool = False

    @property
    def effective_scale(self) -> float:
        return self.device_scale_factor * self.zoom_factor


class Screenshot(BaseModel):
    image_path: str
    image_hash: str
    width_px: int
    height_px: int
    meta: ScreenshotMeta


class ElementType(str, Enum):
    BUTTON = "button"
    TEXT_INPUT = "text_input"
    LINK = "link"
    TEXT = "text"
    ICON = "icon"
    CHECKBOX = "checkbox"
    UNKNOWN = "unknown"


class ElementSource(str, Enum):
    DETECTOR = "detector"
    OCR = "ocr"
    FUSION = "fusion"


class Relation(BaseModel):
    kind: str  # "nearby" | "label_of" | "contained_in"
    target_id: str


class UIElement(BaseModel):
    id: str
    type: ElementType
    bbox: BBox
    text: str | None = None
    confidence: float
    interactable: bool
    semantic_role: str | None = None
    relations: list[Relation] = Field(default_factory=list)
    source: ElementSource


class SemanticUIState(BaseModel):
    elements: list[UIElement]
    screenshot: Screenshot


class ActionOutcome(str, Enum):
    SUCCESS = "success"
    FAILED_NO_TARGET = "failed_no_target"
    FAILED_AMBIGUOUS = "failed_ambiguous"
    FAILED_OUT_OF_VIEWPORT = "failed_out_of_viewport"
    FAILED_EXECUTION_ERROR = "failed_execution_error"


class ActionRecord(BaseModel):
    """Result of one Action Executor step (find/click/type)."""

    action: str  # "find" | "click" | "type"
    target_element_id: str | None = None
    click_point: tuple[float, float] | None = None  # viewport CSS px
    value: str | None = None  # typed text, for "type" actions
    outcome: ActionOutcome
    error_message: str | None = None


class VerificationResult(BaseModel):
    """Result of a pixels-first post-action check — never DOM-based at
    runtime (Instructions.md #2/#5). See visipilot/action/verification.py.
    """

    passed: bool
    method: str  # e.g. "ocr_text_present"
    detail: str | None = None
