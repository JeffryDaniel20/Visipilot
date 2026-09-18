"""Bounded, evaluation-only benchmark for implementation-plan.md B.9:
independently evaluate Moondream2 (`vikhyatk/moondream2`, Apache 2.0) for
textless-icon understanding — deliberately **not** gated on OWLv2
proposing the candidate region first (B.8 already established OWLv2's
own recall on this control class is unreliable). Crops are taken
directly from DOM ground truth (`visipilot/eval/dom_ground_truth.py`,
eval-only, used purely to decide *where to crop for this benchmark*,
never to produce a runtime target) — this isolates "can Moondream2
understand what a textless icon depicts" from "can OWLv2 find it",
which is exactly the point of this milestone.

Real Windows/RTX 5060 loadability, VRAM, and latency are measured, not
estimated. See the module docstring's conclusion and implementation-plan.md
B.9 for the full result: Moondream2 also does not load cleanly against
this project's verified `transformers==5.17.0` without a monkeypatch,
and even once loaded, produces incoherent, repetitive output at
unacceptable latency — a decisive negative result, documented here so
the exact patch and failure mode are reproducible, not just described.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from PIL import Image

from visipilot.capture.screenshot import capture_screenshot, launch_page
from visipilot.eval.dom_ground_truth import get_element_bbox
from visipilot.testserver import TestPageServer

logger = logging.getLogger("visipilot.eval.moondream_bench")

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTPAGES_DIR = REPO_ROOT / "testpages"
DEFAULT_REPORT = REPO_ROOT / "out" / "moondream_bench_report.json"
MOONDREAM_MODEL_ID = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2025-06-21"  # pinned per the model's own usage docs at investigation time

ICON_PAGES = [
    ("search", "search_small_icons.html", "#search-btn", ["search", "magnify", "magnifying", "find", "loupe", "glass"]),
    ("settings", "icon_settings.html", "#settings-btn", ["setting", "gear", "cog", "config", "preferences"]),
    ("menu", "icon_menu.html", "#menu-btn", ["menu", "hamburger", "list", "navigation", "lines"]),
    ("close", "icon_close.html", "#close-btn", ["close", "cross", "cancel", "exit", "x mark", "dismiss"]),
]


def _matches_keywords(caption: str, expected_keywords: list[str]) -> bool:
    return any(kw in caption.lower() for kw in expected_keywords)


def _load_moondream2():
    """Attempt to load Moondream2. Applies the single monkeypatch found
    necessary during investigation: `transformers`' accelerate
    integration (`transformers/integrations/accelerate.py`) directly
    accesses `model.all_tied_weights_keys` without a `getattr` default
    during `from_pretrained`'s device-dispatch path, and Moondream2's
    custom `HfMoondream` remote-code class never sets it (its remote
    code predates this `transformers` 5.x internal attribute). This is
    a workaround for a third-party/library integration gap, not a
    sanctioned fix — documented, not hidden. Unlike Florence-2's 4-layer
    cascade (B.7), this is the *only* patch needed for Moondream2 to
    load without raising — but see the module docstring and B.9 for why
    loading without raising is not the same as loading *correctly*.
    """
    from transformers import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MOONDREAM_MODEL_ID, revision=MOONDREAM_REVISION, trust_remote_code=True, dtype=torch.float16,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model


def run_benchmark(report_path: Path = DEFAULT_REPORT) -> dict:
    logging.basicConfig(level=logging.INFO)

    load_error: str | None = None
    model = None
    load_time_s = None
    t_load = time.perf_counter()
    try:
        model = _load_moondream2()
        load_time_s = time.perf_counter() - t_load
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception as exc:  # noqa: BLE001 — any load failure is real evidence, not swallowed
        load_error = f"{type(exc).__name__}: {exc}"
        logger.error("Moondream2 failed to load: %s", load_error)

    per_page_results = []
    with TestPageServer(directory=str(TESTPAGES_DIR), port=0) as server:
        for name, page_file, sel, expected_keywords in ICON_PAGES:
            url = server.url(page_file)
            with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
                shot = capture_screenshot(page)
                truth_bbox = get_element_bbox(page, sel)

                page_result = {
                    "page": name,
                    "expected_keywords": expected_keywords,
                    "caption": None,
                    "caption_latency_ms": None,
                    "semantic_match": None,
                }

                if model is not None:
                    image = Image.open(shot.image_path).convert("RGB")
                    pad = 4
                    crop = image.crop((
                        max(0, int(truth_bbox.x) - pad), max(0, int(truth_bbox.y) - pad),
                        int(truth_bbox.x2) + pad, int(truth_bbox.y2) + pad,
                    ))
                    try:
                        t0 = time.perf_counter()
                        result = model.caption(crop, length="short")
                        latency_ms = (time.perf_counter() - t0) * 1000
                        caption = result["caption"] if isinstance(result, dict) else str(result)
                        page_result["caption"] = caption
                        page_result["caption_latency_ms"] = round(latency_ms, 1)
                        page_result["semantic_match"] = _matches_keywords(caption, expected_keywords)
                    except Exception as exc:  # noqa: BLE001
                        page_result["caption_error"] = f"{type(exc).__name__}: {exc}"

                per_page_results.append(page_result)

    report = {
        "model_id": MOONDREAM_MODEL_ID,
        "revision": MOONDREAM_REVISION,
        "load_error": load_error,
        "load_time_s": round(load_time_s, 1) if load_time_s else None,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else None,
        "pages": per_page_results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"=== Moondream2 bounded benchmark ({MOONDREAM_MODEL_ID}@{MOONDREAM_REVISION}) ===")
    if load_error:
        print(f"Moondream2 FAILED TO LOAD: {load_error}")
    else:
        print(f"Load time: {load_time_s:.1f}s")
    for r in per_page_results:
        line = f"  {r['page']}: latency_ms={r['caption_latency_ms']} semantic_match={r['semantic_match']} caption={r['caption']!r}"
        print(line.encode("ascii", errors="replace").decode("ascii"))
    print(f"Peak VRAM: {report['peak_vram_mb']}")
    print(f"Report written to {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    run_benchmark(args.report)
