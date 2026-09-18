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
    FAILED_STALE_SCREENSHOT = "failed_stale_screenshot"
    FAILED_RETRY_EXHAUSTED = "failed_retry_exhausted"


class ActionRecord(BaseModel):
    """Result of one Action Executor step (find/click/type).

    One record per *attempt*, not per step: when Phase C's bounded
    re-perception retry re-tries a step, each attempt produces its own
    record, so a trace shows exactly what was tried and why it was
    retried rather than collapsing a retried step into a single
    after-the-fact outcome (Instructions.md's traceability requirement —
    "every retry is observable" is only true if each attempt is recorded).
    """

    action: str  # "find" | "click" | "type"
    target_element_id: str | None = None
    click_point: tuple[float, float] | None = None  # viewport CSS px
    value: str | None = None  # typed text, for "type" actions
    outcome: ActionOutcome
    error_message: str | None = None
    attempt: int = 1  # 1-based; >1 means this step was retried
    reperceived: bool = False  # this attempt ran against freshly re-perceived state
    clarification_used: bool = False  # a clarifier resolved an otherwise-ambiguous target


class RetrySummary(BaseModel):
    """Aggregate retry/clarification counters for one run, derived from
    the per-attempt `ActionRecord`s so the two can never disagree.
    """

    total_attempts: int = 0
    retried_attempts: int = 0  # attempts with attempt > 1
    reperceptions: int = 0
    clarifications_used: int = 0


class VerificationResult(BaseModel):
    """Result of a pixels-first post-action check — never DOM-based at
    runtime (Instructions.md #2/#5). See visipilot/action/verification.py.
    """

    passed: bool
    method: str  # e.g. "ocr_text_present"
    detail: str | None = None


class RuntimeInfo(BaseModel):
    torch_version: str
    cuda_available: bool
    device: str


class TraceRecord(BaseModel):
    """Full record of one end-to-end instruction run, written to disk for
    every run (Instructions.md's traceability requirement /
    implementation-plan.md A.6-A.7).

    Adapted from the A.7 schema sketch to the types actually implemented
    in this project: one record per whole-instruction run (not one per
    action step) since `visipilot.action.runner.run_steps` already
    returns the full per-step `action_records` list for a run, and OCR
    output is a `list[UIElement]` (source=OCR) rather than a separate
    OCRResult type — there's no second schema to keep in sync.
    """

    run_id: str
    instruction: str
    screenshot_hash: str
    detected_elements: list[UIElement]
    ocr_elements: list[UIElement]
    fused_elements: list[UIElement]
    action_records: list[ActionRecord]
    retry_summary: RetrySummary = Field(default_factory=RetrySummary)
    verification: VerificationResult | None = None
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=dict)
    runtime_info: RuntimeInfo
    failure_reason: str | None = None
