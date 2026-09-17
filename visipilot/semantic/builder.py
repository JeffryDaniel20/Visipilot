"""Semantic UI Representation Builder: fuse detector+OCR output, infer
semantic roles and relations, and wrap it all into a SemanticUIState —
the object Target Selection consumes.
"""
from __future__ import annotations

from visipilot.perception.fusion import fuse
from visipilot.semantic.relations import infer_relations
from visipilot.semantic.semantic_role import infer_semantic_role
from visipilot.types import Screenshot, SemanticUIState, UIElement


def build_semantic_state(
    screenshot: Screenshot,
    detector_elements: list[UIElement],
    ocr_elements: list[UIElement],
) -> SemanticUIState:
    fused = fuse(detector_elements, ocr_elements)
    with_roles = [el.model_copy(update={"semantic_role": infer_semantic_role(el)}) for el in fused]
    with_relations = infer_relations(with_roles)
    return SemanticUIState(elements=with_relations, screenshot=screenshot)
