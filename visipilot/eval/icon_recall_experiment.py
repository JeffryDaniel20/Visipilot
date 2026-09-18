"""Bounded, evaluation-only experiment for implementation-plan.md B.8: why
does OWLv2 (the production candidate generator) miss some small, flat,
textless icon controls entirely, and can a reasonable, evidence-based
adjustment (query vocabulary, confidence threshold, image resolution)
fix it without a proportionate increase in false-positive candidate
generation elsewhere?

Eval-only, pixels-first-compliant tooling (screenshot pixels only, no
DOM access except the ground-truth bbox used purely to score results,
exactly like every other eval module in this package) — never imported
by runtime code, and this experiment does not change the shipped
`UIDetector` defaults; see the module docstring's conclusion and
implementation-plan.md B.8 for why.

Usage:
    python -m visipilot.eval.icon_recall_experiment
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from PIL import Image

from visipilot.eval.dom_ground_truth import get_element_bbox
from visipilot.capture.screenshot import capture_screenshot, launch_page
from visipilot.perception.detector import UIDetector
from visipilot.perception.ocr import OCREngine
from visipilot.semantic.builder import build_semantic_state
from visipilot.target_selection.matcher import _NEIGHBOR_MAX_DIMENSION_PX
from visipilot.testserver import TestPageServer
from visipilot.types import ElementType

logger = logging.getLogger("visipilot.eval.icon_recall_experiment")

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTPAGES_DIR = REPO_ROOT / "testpages"
DEFAULT_REPORT = REPO_ROOT / "out" / "icon_recall_experiment_report.json"

# (page, selector-of-the-real-small-icon-target-or-None, note). None
# means the page has no real small-icon target at all -- it exists
# purely to measure false-positive candidate generation on a page that
# should produce zero compact interactable candidates.
PAGES: list[tuple[str, str | None, str]] = [
    ("search_basic.html", None, "no small icon target -- FP exposure only"),
    ("search_bootstrap.html", None, "no small icon target -- FP exposure only"),
    ("search_small_icons.html", "#search-btn", "16x16 icon button"),
    ("search_dark_mode.html", None, "no small icon target -- FP exposure only"),
    ("search_duplicate.html", None, "no small icon target -- FP exposure only"),
    ("search_scroll.html", None, "no small icon target -- FP exposure only"),
    ("icon_settings.html", "#settings-btn", "18x18 gear icon"),
    ("icon_menu.html", "#menu-btn", "18x18 hamburger icon"),
    ("icon_close.html", "#close-btn", "18x18 close icon"),
]

_PRODUCTION_QUERIES = {
    "a rectangular text input box": ElementType.TEXT_INPUT,
    "a button": ElementType.BUTTON,
}
_ICON_QUERIES = {
    **_PRODUCTION_QUERIES,
    "a small icon button": ElementType.ICON,
    "a toolbar icon button": ElementType.ICON,
}

QUERY_CONFIGS = {
    "production_baseline": {"queries": _PRODUCTION_QUERIES, "threshold": 0.1},
    "icon_queries_t010": {"queries": _ICON_QUERIES, "threshold": 0.1},
    "icon_queries_t005": {"queries": _ICON_QUERIES, "threshold": 0.05},
    "baseline_lower_threshold_t005": {"queries": _PRODUCTION_QUERIES, "threshold": 0.05},
}


def _overlap_ratio(bbox, truth) -> float:
    ix1, iy1 = max(bbox.x, truth.x), max(bbox.y, truth.y)
    ix2, iy2 = min(bbox.x2, truth.x2), min(bbox.y2, truth.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    truth_area = truth.width * truth.height
    return inter / truth_area if truth_area else 0.0


def _is_compact_candidate(el) -> bool:
    """Same narrowing B.5/B.7 use: compact, textless, interactable."""
    return (
        el.interactable
        and not el.text
        and el.bbox.width <= _NEIGHBOR_MAX_DIMENSION_PX
        and el.bbox.height <= _NEIGHBOR_MAX_DIMENSION_PX
    )


def run_experiment(report_path: Path = DEFAULT_REPORT) -> dict:
    ocr = OCREngine(gpu=True)
    results: dict = {}

    with TestPageServer(directory=str(TESTPAGES_DIR), port=0) as server:
        page_cache = {}
        for page_name, sel, note in PAGES:
            url = server.url(page_name)
            with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
                shot = capture_screenshot(page)
                truth = get_element_bbox(page, sel) if sel else None
                ocr_els = ocr.read(shot.image_path)
                page_cache[page_name] = (shot, truth, ocr_els, note)

        for config_name, cfg in QUERY_CONFIGS.items():
            detector = UIDetector(queries=cfg["queries"], score_threshold=cfg["threshold"])
            config_result: dict = {"pages": {}, "total_recall_targets": 0, "total_recalled": 0, "total_fp_candidates": 0}
            for page_name, sel, note in PAGES:
                shot, truth, ocr_els, note = page_cache[page_name]
                det = detector.detect(shot.image_path)
                state = build_semantic_state(shot, det, ocr_els)
                candidates = [el for el in state.elements if _is_compact_candidate(el)]

                recalled = False
                best_overlap = 0.0
                fp_count = 0
                for el in candidates:
                    ov = _overlap_ratio(el.bbox, truth) if truth else 0.0
                    best_overlap = max(best_overlap, ov)
                    if truth and ov >= 0.5:
                        recalled = True
                    else:
                        fp_count += 1

                config_result["pages"][page_name] = {
                    "note": note,
                    "num_compact_candidates": len(candidates),
                    "best_overlap_with_truth": round(best_overlap, 3) if truth else None,
                    "recalled": recalled if truth else None,
                    "fp_candidates": fp_count,
                }
                if truth:
                    config_result["total_recall_targets"] += 1
                    config_result["total_recalled"] += int(recalled)
                config_result["total_fp_candidates"] += fp_count
            results[config_name] = config_result

        # Separate, smaller probe: does 2x upscaling raise confidence at
        # the true icon location on the two pages the production
        # baseline misses, independent of query/threshold changes?
        upscale_probe = {}
        low_conf_detector = UIDetector(score_threshold=0.01)
        for page_name, sel, note in PAGES:
            if sel not in ("#settings-btn", "#menu-btn"):
                continue
            shot, truth, _, _ = page_cache[page_name]
            img = Image.open(shot.image_path)
            upscaled_path = str(shot.image_path).replace(".png", "_2x.png")
            img.resize((img.width * 2, img.height * 2), Image.LANCZOS).save(upscaled_path)
            det_2x = low_conf_detector.detect(upscaled_path)
            best_conf_at_truth = 0.0
            for el in det_2x:
                b = el.bbox
                orig_bbox_x2 = b.x / 2 + b.width / 2
                # map back to original coords for overlap scoring
                from visipilot.types import BBox
                orig_bbox = BBox(x=b.x / 2, y=b.y / 2, width=b.width / 2, height=b.height / 2, space=b.space)
                if _overlap_ratio(orig_bbox, truth) >= 0.5:
                    best_conf_at_truth = max(best_conf_at_truth, el.confidence)
            upscale_probe[page_name] = {"best_confidence_2x_upscaled": round(best_conf_at_truth, 4)}

    results["upscale_probe_at_true_location"] = upscale_probe

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("=== OWLv2 icon-recall experiment ===")
    for config_name, cfg_result in results.items():
        if config_name == "upscale_probe_at_true_location":
            continue
        print(f"  {config_name}: recall={cfg_result['total_recalled']}/{cfg_result['total_recall_targets']} "
              f"total_fp_candidates={cfg_result['total_fp_candidates']}")
    print(f"  upscale probe (raw confidence at true location, 2x image): {upscale_probe}")
    print(f"Report written to {report_path}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_experiment()
