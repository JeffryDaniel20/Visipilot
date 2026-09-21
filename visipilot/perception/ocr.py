"""OCR engine: EasyOCR wrapper producing text + bounding boxes from
screenshot pixels.

Chosen over PaddleOCR for this first integration (Instructions.md #2/#10
decision rules): EasyOCR is PyTorch-native, reusing the CUDA/Blackwell
compatibility already verified in Phase 0, instead of adding PaddlePaddle
as a second deep-learning framework with its own unverified Blackwell
support. PaddleOCR remains the documented fallback if EasyOCR's accuracy
proves insufficient on dense/small UI text (see implementation-plan.md).
License: Apache 2.0.
"""
from __future__ import annotations

from pathlib import Path

from visipilot.types import BBox, CoordinateSpace, ElementSource, ElementType, UIElement


class OCREngine:
    """Wraps EasyOCR. Loads models once at construction; reuse one
    instance across multiple read() calls to amortize load cost.
    """

    def __init__(self, languages: list[str] | None = None, gpu: bool = True):
        import easyocr

        self.languages = languages or ["en"]
        self.gpu = gpu
        self.reader = easyocr.Reader(self.languages, gpu=gpu, verbose=False)

    def read(self, image_path: str | Path, batch_size: int | None = None) -> list[UIElement]:
        """`batch_size`, when given, is passed straight through to
        EasyOCR's own `readtext(batch_size=...)` — the number of
        detected text-region crops recognized together per GPU call
        (EasyOCR's own default, used when omitted here, is 1: fully
        serial). Measured directly (not assumed): raising it changes
        neither the recognized text on any region nor VRAM usage
        meaningfully, only throughput — see the caller in
        `visipilot/tracing/pipeline.py` for why this is only ever passed
        for the very first, pre-action perception call, never for a
        verification or re-perception check (implementation-plan.md
        C.9's "do not touch C.5" finding).
        """
        results = self.reader.readtext(str(image_path), batch_size=batch_size or 1)

        elements: list[UIElement] = []
        for i, (points, text, confidence) in enumerate(results):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            elements.append(
                UIElement(
                    id=f"ocr-{i}",
                    type=ElementType.TEXT,
                    bbox=BBox(
                        x=x1,
                        y=y1,
                        width=x2 - x1,
                        height=y2 - y1,
                        space=CoordinateSpace.SCREENSHOT_PX,
                    ),
                    text=text,
                    confidence=float(confidence),
                    interactable=False,
                    source=ElementSource.OCR,
                )
            )
        return elements
