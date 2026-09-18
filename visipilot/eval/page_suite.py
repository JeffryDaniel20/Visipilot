"""Phase B robustness suite: runs `visipilot.eval.vertical_slice.run_batch()`
across multiple, meaningfully different controlled test pages, reusing
one loaded detector/OCR pair for the whole suite (loading them once,
not once per page) and reports per-page + suite-level results.

Deliberately reuses `run_batch()` entirely rather than duplicating its
logic — the per-page differences that matter (layout, styling, small
elements, dark mode, duplicate targets, below-the-fold placement) are
expressed as page content and PageSpec config, not new pipeline code.

Usage:
    python -m visipilot.eval.page_suite --runs-per-page 5 --report out/page_suite_report.json

Each PageSpec's `note` documents exactly what real-world condition it's
meant to stress — see implementation-plan.md Phase B for why each one
was chosen and what was actually observed.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from visipilot.eval.vertical_slice import DEFAULT_INSTRUCTION, REPO_ROOT, run_batch
from visipilot.perception.detector import UIDetector
from visipilot.perception.ocr import OCREngine

logger = logging.getLogger("visipilot.eval.page_suite")

TESTPAGES_DIR = REPO_ROOT / "testpages"
DEFAULT_REPORT = REPO_ROOT / "out" / "page_suite_report.json"
DEFAULT_RUNS_PER_PAGE = 5  # see implementation-plan.md Phase B for why 5, not 20


@dataclass
class PageSpec:
    name: str
    page_path: Path
    note: str  # what real-world condition this page is stressing
    instruction: str = DEFAULT_INSTRUCTION
    verify_text: str | None = "Results for: Python"
    ground_truth_selectors: dict[str, str] | None = None
    full_page: bool = False

    def __post_init__(self):
        if self.ground_truth_selectors is None:
            self.ground_truth_selectors = {"input": "#search-box", "button": "#search-btn"}


PAGE_SUITE: list[PageSpec] = [
    PageSpec(
        name="basic",
        page_path=TESTPAGES_DIR / "search_basic.html",
        note="Phase A baseline — regression check, not a new condition.",
    ),
    PageSpec(
        name="bootstrap_style",
        page_path=TESTPAGES_DIR / "search_bootstrap.html",
        note="Different CSS framework conventions: pill/rounded input, "
             "Bootstrap-blue #0d6efd button, different font stack and spacing.",
    ),
    PageSpec(
        name="small_icon_button",
        page_path=TESTPAGES_DIR / "search_small_icons.html",
        note="Sub-20px (16x16) icon-only search button with NO text label — "
             "stresses small-element detection and text-free matching.",
    ),
    PageSpec(
        name="dark_mode",
        page_path=TESTPAGES_DIR / "search_dark_mode.html",
        note="Dark background, dark (not white) input fill — directly "
             "stresses the matcher's documented fill-color-from-white assumption.",
    ),
    PageSpec(
        name="duplicate_search_boxes",
        page_path=TESTPAGES_DIR / "search_duplicate.html",
        note="Two structurally/visually identical search box+button pairs — "
             "stresses ambiguity handling under harder real conditions.",
        ground_truth_selectors={"input": "#search-box", "button": "#search-btn"},
    ),
    PageSpec(
        name="scroll_below_fold",
        page_path=TESTPAGES_DIR / "search_scroll.html",
        note="Search box ~1900px below an 800px viewport — stresses "
             "whether perception sees off-screen content and whether the "
             "action layer correctly refuses to click outside the viewport.",
        full_page=True,
    ),
    PageSpec(
        name="dynamic_content",
        page_path=TESTPAGES_DIR / "search_dynamic.html",
        note="A banner appears ~200ms after load, shifting the search row "
             "down — stresses Instructions.md #7's stale-screenshot rule: "
             "does the real detector+OCR latency window let the page "
             "change before the first CLICK/TYPE action, and does the "
             "runner refuse rather than click now-invalid coordinates?",
    ),
    PageSpec(
        name="occluded_button",
        page_path=TESTPAGES_DIR / "search_occluded.html",
        note="A small sibling badge overlays the search button's top-right "
             "corner (its clickable center is unaffected) — stresses "
             "whether partial visual occlusion distorts detection/fusion "
             "for an otherwise normal button.",
    ),
]


def run_suite(
    runs_per_page: int = DEFAULT_RUNS_PER_PAGE,
    report_path: Path = DEFAULT_REPORT,
    device_scale_factor: float = 1.0,
) -> dict:
    detector = UIDetector()
    ocr = OCREngine(gpu=True)
    torch.cuda.reset_peak_memory_stats()

    per_page: dict[str, dict] = {}
    for spec in PAGE_SUITE:
        logger.info("=== page=%s note=%r dpr=%s ===", spec.name, spec.note, device_scale_factor)
        summary = run_batch(
            page_path=spec.page_path,
            instruction=spec.instruction,
            runs=runs_per_page,
            verify_text=spec.verify_text,
            ground_truth_selectors=spec.ground_truth_selectors,
            detector=detector,
            ocr=ocr,
            full_page=spec.full_page,
            min_runs_for_gate=runs_per_page,
            device_scale_factor=device_scale_factor,
        )
        per_page[spec.name] = {"note": spec.note, "full_page_capture": spec.full_page, **summary}
        logger.info(
            "=== page=%s done: success_rate=%.0f%% click_accuracy=%s max_latency_s=%.2f ===",
            spec.name, summary["success_rate"] * 100, summary["click_accuracy"], summary["max_latency_s"],
        )

    suite_peak_vram_mb = max(p["peak_vram_mb"] for p in per_page.values())
    suite_peak_ram_mb = max(p["peak_ram_mb"] for p in per_page.values())

    report = {
        "runs_per_page": runs_per_page,
        "pages": list(per_page.keys()),
        "suite_peak_vram_mb": suite_peak_vram_mb,
        "suite_peak_ram_mb": suite_peak_ram_mb,
        "per_page": per_page,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _print_report(report: dict, report_path: Path) -> None:
    print(f"\n=== Phase B page-suite evaluation: {len(report['pages'])} pages x {report['runs_per_page']} runs ===")
    for name, summary in report["per_page"].items():
        click_acc = summary["click_accuracy"]
        click_acc_str = f"{click_acc*100:.0f}%" if click_acc is not None else "n/a"
        print(
            f"  {name:24s} success={summary['successes']}/{summary['runs']} "
            f"click_acc={click_acc_str:>5s} max_latency={summary['max_latency_s']:.2f}s "
            f"peak_vram={summary['peak_vram_mb']:.0f}MB"
        )
        if summary["failures"]:
            for f in summary["failures"]:
                print(f"      FAILURE run {f['run_index']}: {f['failure_reason']}")
    print(f"\nSuite peak VRAM: {report['suite_peak_vram_mb']:.1f} MB")
    print(f"Suite peak RAM: {report['suite_peak_ram_mb']:.1f} MB")
    print(f"Report written to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VisiPilot Phase B page-suite evaluation")
    parser.add_argument("--runs-per-page", type=int, default=DEFAULT_RUNS_PER_PAGE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dpr", type=float, default=1.0, help="device_scale_factor for the whole suite (Phase B DPR sweep)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = run_suite(runs_per_page=args.runs_per_page, report_path=args.report, device_scale_factor=args.dpr)
    _print_report(report, args.report)


if __name__ == "__main__":
    main()
