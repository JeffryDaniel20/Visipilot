"""UI element detector: OWLv2 zero-shot open-vocabulary object detection.

Model: google/owlv2-base-patch16-ensemble (Apache 2.0), via Hugging Face
transformers. Chosen over OmniParser's icon_detect (AGPL-3.0) to keep the
project's licensing clean by default (Instructions.md #2), and chosen
over Grounding DINO for this first integration because it has first-class
`transformers` support and a smaller footprint. See implementation-plan.md
for the accuracy this actually achieves on the controlled test page —
OWLv2 was trained on natural images, not UI screenshots, so this is a
deliberately-measured starting point, not an assumed-good fit.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from visipilot.types import BBox, CoordinateSpace, ElementSource, ElementType, UIElement

DEFAULT_MODEL_ID = "google/owlv2-base-patch16-ensemble"

# Open-vocabulary text queries mapped to our ElementType taxonomy.
DEFAULT_QUERIES: dict[str, ElementType] = {
    "a rectangular text input box": ElementType.TEXT_INPUT,
    "a button": ElementType.BUTTON,
}

_INTERACTABLE_TYPES = {
    ElementType.BUTTON,
    ElementType.TEXT_INPUT,
    ElementType.ICON,
    ElementType.LINK,
    ElementType.CHECKBOX,
}


class UIDetector:
    """Wraps OWLv2 for zero-shot UI element detection.

    Loads the model once at construction; reuse one instance across
    multiple detect() calls to amortize load cost (measured separately
    from per-call inference latency — see TESTING.md).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        queries: dict[str, ElementType] | None = None,
        score_threshold: float = 0.1,
    ):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.queries = queries or DEFAULT_QUERIES
        self.score_threshold = score_threshold
        self.model_id = model_id
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def detect(self, image_path: str | Path) -> list[UIElement]:
        image = Image.open(image_path).convert("RGB")
        query_texts = list(self.queries.keys())

        inputs = self.processor(text=[query_texts], images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.score_threshold,
            text_labels=[query_texts],
        )[0]
        # Match on the returned text string, not the numeric `labels`
        # index — `labels` indexes OWLv2's internal per-box class-prior
        # ordering, not the position in our query list, and silently
        # produced wrong types when trusted directly (verified by
        # inspecting `text_labels` against `labels` during development).
        matched_texts = results["text_labels"]

        elements: list[UIElement] = []
        for i in range(len(results["scores"])):
            x1, y1, x2, y2 = [float(v) for v in results["boxes"][i].tolist()]
            score = float(results["scores"][i])
            query_text = matched_texts[i]
            element_type = self.queries[query_text]
            elements.append(
                UIElement(
                    id=f"det-{i}",
                    type=element_type,
                    bbox=BBox(
                        x=x1,
                        y=y1,
                        width=x2 - x1,
                        height=y2 - y1,
                        space=CoordinateSpace.SCREENSHOT_PX,
                    ),
                    confidence=score,
                    interactable=element_type in _INTERACTABLE_TYPES,
                    source=ElementSource.DETECTOR,
                )
            )
        return elements
