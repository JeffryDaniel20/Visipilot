"""Bounded evaluation-only benchmark for implementation-plan.md B.6/B.7:
does Florence-2-base's region-captioning task produce a caption that
*semantically matches* an icon-only control's real meaning (not just
"some icon was detected")?

This is deliberately eval-only tooling, never imported by runtime
pipeline code, and never itself part of the runtime action path — per
this milestone's explicit instruction, Florence-2 is NOT wired into
`match_target()`/the runner here. It runs the existing, already-verified
perception pipeline (OWLv2 detector + EasyOCR) exactly as the runtime
does, reuses B.5's candidate-narrowing predicate (compact + textless +
interactable) to pick the crop regions to caption, then attempts a
Florence-2-base caption on each crop and scores it against a small
per-page keyword set.

Known, currently-unresolved blocker (documented, not swallowed): as of
this milestone, `microsoft/Florence-2-base`'s official HF remote code
does not load correctly against `transformers==5.17.0` (this project's
already-verified, currency-checked pin) without unofficial monkeypatches
to Microsoft's own remote code, and even once coerced to load, its
language-model embedding/lm_head weights fail to load from the checkpoint
(reported by transformers' own load report as "MISSING ... newly
initialized"), producing incoherent captions unrelated to the input
image. See implementation-plan.md B.7 for the full investigation and the
public issue this corroborates (huggingface/transformers#36886 — Florence-2
broke starting transformers 4.50.0, unresolved by Microsoft as of this
writing). This module still implements the real captioning path (not a
stub) so it can be re-run as-is the moment that upstream incompatibility
is resolved, without rewriting the benchmark.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

from visipilot.capture.screenshot import capture_screenshot, launch_page
from visipilot.eval.dom_ground_truth import get_element_bbox
from visipilot.perception.detector import UIDetector
from visipilot.perception.ocr import OCREngine
from visipilot.semantic.builder import build_semantic_state
from visipilot.target_selection.matcher import _NEIGHBOR_MAX_DIMENSION_PX
from visipilot.testserver import TestPageServer
from visipilot.types import UIElement

logger = logging.getLogger("visipilot.eval.icon_caption_bench")

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTPAGES_DIR = REPO_ROOT / "testpages"
DEFAULT_REPORT = REPO_ROOT / "out" / "icon_caption_bench_report.json"
FLORENCE_MODEL_ID = "microsoft/Florence-2-base"


@dataclass
class IconPageSpec:
    name: str
    page_path: Path
    btn_selector: str
    expected_keywords: list[str]  # any one matching = semantic pass


ICON_PAGES: list[IconPageSpec] = [
    IconPageSpec("search", TESTPAGES_DIR / "search_small_icons.html", "#search-btn", ["search", "magnify", "magnifying", "find", "loupe", "glass"]),
    IconPageSpec("settings", TESTPAGES_DIR / "icon_settings.html", "#settings-btn", ["setting", "gear", "cog", "config", "preferences"]),
    IconPageSpec("menu", TESTPAGES_DIR / "icon_menu.html", "#menu-btn", ["menu", "hamburger", "list", "navigation", "lines"]),
    IconPageSpec("close", TESTPAGES_DIR / "icon_close.html", "#close-btn", ["close", "cross", "cancel", "exit", "x mark", "dismiss"]),
]


def _is_icon_candidate(el: UIElement) -> bool:
    """Same narrowing B.5 already uses for the redirect: compact,
    textless, interactable — not a new heuristic, reused deliberately so
    this benchmark tests captioning exactly the candidates the runtime
    pipeline would actually caption if this were ever wired in.
    """
    return (
        el.interactable
        and not el.text
        and el.bbox.width <= _NEIGHBOR_MAX_DIMENSION_PX
        and el.bbox.height <= _NEIGHBOR_MAX_DIMENSION_PX
    )


def _load_florence2():
    """Attempt to load Florence-2-base. Applies the minimum monkeypatches
    found necessary during investigation to get past its remote code's
    incompatibilities with transformers 5.17.0's config/tokenizer APIs —
    documented, not hidden, and clearly labeled as workarounds for a
    third party's code, not sanctioned upstream fixes. Raises with a
    clear message if loading still fails (e.g. the deeper
    generation-cache/weight-loading issues found this milestone), rather
    than silently returning None.
    """
    from transformers import PretrainedConfig, PreTrainedTokenizerBase

    _orig_cfg_getattr = PretrainedConfig.__getattribute__

    def _patched_cfg_getattr(self, key):
        try:
            return _orig_cfg_getattr(self, key)
        except AttributeError:
            if key in ("forced_bos_token_id", "forced_eos_token_id", "bos_token_id", "eos_token_id", "pad_token_id"):
                return None
            raise

    PretrainedConfig.__getattribute__ = _patched_cfg_getattr

    _orig_tok_getattr = PreTrainedTokenizerBase.__getattr__

    def _patched_tok_getattr(self, key):
        if key == "additional_special_tokens":
            return []
        return _orig_tok_getattr(self, key)

    PreTrainedTokenizerBase.__getattr__ = _patched_tok_getattr

    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(FLORENCE_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="eager"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return processor, model


def _caption_crop(processor, model, crop: Image.Image) -> tuple[str, float]:
    device = next(model.parameters()).device
    inputs = processor(text="<CAPTION>", images=crop, return_tensors="pt").to(device, torch.float16)
    t0 = time.perf_counter()
    with torch.inference_mode():
        gen = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=32, num_beams=1, use_cache=False,
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    text = processor.batch_decode(gen, skip_special_tokens=True)[0]
    return text, latency_ms


def run_benchmark(report_path: Path = DEFAULT_REPORT) -> dict:
    logging.basicConfig(level=logging.INFO)
    detector = UIDetector()
    ocr = OCREngine(gpu=True)

    florence_load_error: str | None = None
    processor = model = None
    try:
        processor, model = _load_florence2()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception as exc:  # noqa: BLE001 — deliberately broad: any load failure is real evidence, not swallowed
        florence_load_error = f"{type(exc).__name__}: {exc}"
        logger.error("Florence-2-base failed to load: %s", florence_load_error)

    per_page_results = []
    with TestPageServer(directory=str(TESTPAGES_DIR), port=0) as server:
        for spec in ICON_PAGES:
            url = server.url(spec.page_path.name)
            with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
                shot = capture_screenshot(page)
                truth_bbox = get_element_bbox(page, spec.btn_selector)
                det = detector.detect(shot.image_path)
                ocr_els = ocr.read(shot.image_path)
                state = build_semantic_state(shot, det, ocr_els)

                candidates = [el for el in state.elements if _is_icon_candidate(el)]
                # Score each candidate against DOM ground truth (eval-only)
                # so we caption the one that actually corresponds to the
                # real control, not an arbitrary compact textless box.
                best = None
                best_overlap = 0.0
                for el in candidates:
                    b = el.bbox
                    ix1, iy1 = max(b.x, truth_bbox.x), max(b.y, truth_bbox.y)
                    ix2, iy2 = min(b.x2, truth_bbox.x2), min(b.y2, truth_bbox.y2)
                    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                    inter = iw * ih
                    truth_area = truth_bbox.width * truth_bbox.height
                    overlap = inter / truth_area if truth_area else 0.0
                    if overlap > best_overlap:
                        best, best_overlap = el, overlap

                page_result = {
                    "page": spec.name,
                    "candidates_found": len(candidates),
                    "best_candidate_overlap_with_truth": round(best_overlap, 3),
                    "detector_localized_true_icon": best_overlap >= 0.5,
                    "caption": None,
                    "caption_latency_ms": None,
                    "semantic_match": None,
                    "expected_keywords": spec.expected_keywords,
                }

                if best is not None and processor is not None and model is not None:
                    image = Image.open(shot.image_path).convert("RGB")
                    pad = 4
                    crop_box = (
                        max(0, int(best.bbox.x) - pad), max(0, int(best.bbox.y) - pad),
                        int(best.bbox.x2) + pad, int(best.bbox.y2) + pad,
                    )
                    crop = image.crop(crop_box)
                    try:
                        caption, latency_ms = _caption_crop(processor, model, crop)
                        page_result["caption"] = caption
                        page_result["caption_latency_ms"] = round(latency_ms, 1)
                        page_result["semantic_match"] = any(kw in caption.lower() for kw in spec.expected_keywords)
                    except Exception as exc:  # noqa: BLE001
                        page_result["caption_error"] = f"{type(exc).__name__}: {exc}"

                per_page_results.append(page_result)

    report = {
        "florence_model_id": FLORENCE_MODEL_ID,
        "florence_load_error": florence_load_error,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else None,
        "pages": per_page_results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"=== Icon-caption bounded benchmark ({FLORENCE_MODEL_ID}) ===")
    if florence_load_error:
        print(f"Florence-2-base FAILED TO LOAD: {florence_load_error}")
        print("Captioning was skipped for all pages; detector localization was still measured.")
    for r in per_page_results:
        # Florence-2's garbage output (see the module docstring) can
        # contain characters the Windows console's default codepage
        # can't encode; the JSON report (UTF-8) is the source of truth,
        # this print is a best-effort summary only.
        line = (f"  {r['page']}: candidates={r['candidates_found']} "
                f"localized_true_icon={r['detector_localized_true_icon']} "
                f"caption={r['caption']!r} semantic_match={r['semantic_match']}")
        print(line.encode("ascii", errors="replace").decode("ascii"))
    print(f"Report written to {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    run_benchmark(args.report)
