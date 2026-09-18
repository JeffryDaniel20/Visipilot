"""Vertical-slice evaluation harness: runs the fixed Phase A instruction
N times against the controlled test page, scores each run against DOM
ground truth (eval-only — this module reads the DOM purely to score
results, exactly like `dom_ground_truth.py`; the runtime pipeline it
calls via `visipilot.tracing.pipeline.run_instruction` never does),
and reports pass/fail against every criterion in implementation-plan.md's
"Vertical-slice pass criteria (Phase A)".

Usage:
    python -m visipilot.eval.vertical_slice \
        --page testpages/search_basic.html \
        --instruction "Find the search box, type Python, and click Search." \
        --runs 20 \
        --report out/vertical_slice_report.json

One browser (and one detector/OCR model pair) is reused across all runs
— matching the module-scoped-fixture pattern already used throughout
this project's own integration tests — with a fresh `page.goto(url)`
navigation before each run to reset in-page JS state (the `#results`
div, the input's typed value). This is a normal page load, not a DOM
manipulation, so it doesn't touch the pixels-first rule.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import psutil
import torch
from playwright.sync_api import sync_playwright

from visipilot.eval.dom_ground_truth import GroundTruthElementNotFound, get_element_bbox
from visipilot.perception.detector import UIDetector
from visipilot.perception.ocr import OCREngine
from visipilot.testserver import TestPageServer
from visipilot.tracing.pipeline import DEFAULT_TRACE_DIR, run_instruction
from visipilot.types import BBox

logger = logging.getLogger("visipilot.eval.vertical_slice")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PAGE = REPO_ROOT / "testpages" / "search_basic.html"
DEFAULT_REPORT = REPO_ROOT / "out" / "vertical_slice_report.json"
DEFAULT_INSTRUCTION = "Find the search box, type Python, and click Search."

# Pass criteria — see implementation-plan.md "Vertical-slice pass criteria
# (Phase A)" for the reasoning behind each number. Not to be loosened to
# force a pass; a failing run here is real information, not a bug in the
# harness.
MIN_RUNS_FOR_GATE = 20
MIN_SUCCESS_RATE = 0.90
MIN_CLICK_ACCURACY = 0.95
MAX_LATENCY_S = 5.0
MAX_PEAK_VRAM_MB = 6144.0
MAX_PEAK_RAM_MB = 12288.0

# Maps an ActionRecord.action value to the ground-truth selector name its
# grounded click_point should land inside. Page-specific by design — this
# harness targets the one controlled Phase A page; Phase B's multi-page
# suite will need a per-page mapping, not a hardcoded one.
GROUND_TRUTH_SELECTORS = {"input": "#search-box", "button": "#search-btn"}
CLICK_TARGET_FOR_ACTION = {"click": "button", "type": "input"}


def _click_lands_in_bbox(click_point: tuple[float, float] | None, bbox: BBox | None) -> bool | None:
    """None means "not checkable" (missing click point or ground truth),
    not a failure — callers must not silently count that as a pass.
    """
    if click_point is None or bbox is None:
        return None
    x, y = click_point
    return bbox.x <= x <= bbox.x2 and bbox.y <= y <= bbox.y2


def run_batch(
    page_path: Path,
    instruction: str,
    runs: int,
    verify_text: str | None,
    ground_truth_selectors: dict[str, str] = GROUND_TRUTH_SELECTORS,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    detector: UIDetector | None = None,
    ocr: OCREngine | None = None,
    full_page: bool = False,
    min_runs_for_gate: int = MIN_RUNS_FOR_GATE,
) -> dict:
    """Runs `instruction` `runs` times against `page_path`, returns the
    aggregated summary dict (also what gets written to the report file).

    `detector`/`ocr`, when given, are reused as-is instead of being
    constructed fresh — lets a multi-page suite (visipilot/eval/page_suite.py)
    load the models once and reuse them across every page, instead of
    paying ~50s of model-load time per page. When not given (the
    original single-page CLI behavior), both are constructed here,
    unchanged from before.

    `min_runs_for_gate` defaults to the Phase A vertical-slice gate's 20,
    but is parameterized so Phase B's smaller per-page sample sizes can
    report a meaningful `criteria.min_runs_met` for the N actually run,
    rather than always failing that one check for a deliberately smaller
    sample. This does not change the success-rate/click-accuracy/latency/
    VRAM/RAM thresholds themselves.
    """
    owns_models = detector is None and ocr is None
    if detector is None:
        detector = UIDetector()
    if ocr is None:
        ocr = OCREngine(gpu=True)

    # Reset peak stats AFTER loading the models, matching how every prior
    # resource measurement in this project reports peak VRAM including
    # the loaded-model footprint, not just incremental inference — see
    # implementation-plan.md A.2/A.6 for the numbers this is consistent with.
    # When models are reused across a suite, the caller resets stats once
    # up front instead (resetting here would hide earlier pages' cost).
    if owns_models:
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    peak_rss_mb = process.memory_info().rss / 1024**2

    run_results: list[dict] = []

    with TestPageServer(directory=page_path.resolve().parent, port=0) as server:
        url = server.url(page_path.name)

        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome")
            context = browser.new_context(viewport={"width": 1280, "height": 800}, device_scale_factor=1.0)
            page = context.new_page()

            for i in range(runs):
                page.goto(url)

                t0 = time.perf_counter()
                record = run_instruction(
                    page, instruction, detector, ocr,
                    verify_expected_text=verify_text, trace_dir=trace_dir,
                    full_page=full_page,
                )
                wall_s = time.perf_counter() - t0

                peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / 1024**2)

                ground_truth: dict[str, BBox | None] = {}
                for name, selector in ground_truth_selectors.items():
                    try:
                        ground_truth[name] = get_element_bbox(page, selector)
                    except GroundTruthElementNotFound:
                        ground_truth[name] = None

                click_checks: list[bool] = []
                for action_record in record.action_records:
                    target_key = CLICK_TARGET_FOR_ACTION.get(action_record.action)
                    if target_key is None:
                        continue
                    result = _click_lands_in_bbox(action_record.click_point, ground_truth.get(target_key))
                    if result is not None:
                        click_checks.append(result)

                success = record.failure_reason is None
                run_results.append({
                    "run_index": i,
                    "run_id": record.run_id,
                    "success": success,
                    "failure_reason": record.failure_reason,
                    "wall_time_s": wall_s,
                    "click_checks": click_checks,
                    "click_accuracy_ok": all(click_checks) if click_checks else None,
                    "action_outcomes": [a.outcome.value for a in record.action_records],
                    "verification_passed": record.verification.passed if record.verification else None,
                })
                logger.info(
                    "run %d/%d: success=%s wall_time_s=%.2f click_checks=%s",
                    i + 1, runs, success, wall_s, click_checks,
                )

            context.close()
            browser.close()

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    return _summarize(run_results, peak_vram_mb, peak_rss_mb, min_runs_for_gate=min_runs_for_gate)


def _summarize(run_results: list[dict], peak_vram_mb: float, peak_ram_mb: float, min_runs_for_gate: int = MIN_RUNS_FOR_GATE) -> dict:
    n = len(run_results)
    successes = sum(1 for r in run_results if r["success"])
    success_rate = successes / n if n else 0.0

    click_checked = [r for r in run_results if r["click_accuracy_ok"] is not None]
    click_correct = sum(1 for r in click_checked if r["click_accuracy_ok"])
    click_accuracy = (click_correct / len(click_checked)) if click_checked else None

    latencies = [r["wall_time_s"] for r in run_results]
    max_latency = max(latencies) if latencies else 0.0
    mean_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

    criteria = {
        "min_runs_met": n >= min_runs_for_gate,
        "success_rate_ok": success_rate >= MIN_SUCCESS_RATE,
        # No click actions being checkable at all is a harness/page-config
        # problem, not a pass — never let None silently satisfy the gate.
        "click_accuracy_ok": click_accuracy is not None and click_accuracy >= MIN_CLICK_ACCURACY,
        "latency_ok": max_latency <= MAX_LATENCY_S,
        "vram_ok": peak_vram_mb <= MAX_PEAK_VRAM_MB,
        "ram_ok": peak_ram_mb <= MAX_PEAK_RAM_MB,
    }
    overall_pass = n >= min_runs_for_gate and all(criteria.values())

    return {
        "runs": n,
        "successes": successes,
        "success_rate": success_rate,
        "click_accuracy": click_accuracy,
        "click_checked_count": len(click_checked),
        "max_latency_s": max_latency,
        "mean_latency_s": mean_latency,
        "peak_vram_mb": peak_vram_mb,
        "peak_ram_mb": peak_ram_mb,
        "criteria": criteria,
        "overall_pass": overall_pass,
        "failures": [r for r in run_results if not r["success"]],
        "run_results": run_results,
    }


def _print_report(summary: dict, report_path: Path) -> None:
    def status(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(f"\n=== Vertical-slice evaluation: {summary['runs']} runs ===")
    print(
        f"Success rate: {summary['successes']}/{summary['runs']} "
        f"({summary['success_rate']*100:.1f}%) — need >= {MIN_SUCCESS_RATE*100:.0f}%: "
        f"{status(summary['criteria']['success_rate_ok'])}"
    )
    if summary["click_accuracy"] is not None:
        print(
            f"Click accuracy: {summary['click_accuracy']*100:.1f}% "
            f"({summary['click_checked_count']} checks) — need >= {MIN_CLICK_ACCURACY*100:.0f}%: "
            f"{status(summary['criteria']['click_accuracy_ok'])}"
        )
    else:
        print("Click accuracy: no click actions were checkable — FAIL (see criteria.click_accuracy_ok)")
    print(
        f"Max latency: {summary['max_latency_s']:.2f}s (mean {summary['mean_latency_s']:.2f}s) "
        f"— need <= {MAX_LATENCY_S}s: {status(summary['criteria']['latency_ok'])}"
    )
    print(f"Peak VRAM: {summary['peak_vram_mb']:.1f} MB — need <= {MAX_PEAK_VRAM_MB:.0f} MB: {status(summary['criteria']['vram_ok'])}")
    print(f"Peak RAM: {summary['peak_ram_mb']:.1f} MB — need <= {MAX_PEAK_RAM_MB:.0f} MB: {status(summary['criteria']['ram_ok'])}")
    print(f"\nOVERALL: {status(summary['overall_pass'])}")
    print(f"Report written to {report_path}")

    if summary["failures"]:
        print(f"\n{len(summary['failures'])} failure(s):")
        for f in summary["failures"]:
            print(f"  run {f['run_index']} ({f['run_id']}): {f['failure_reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VisiPilot vertical-slice evaluation harness")
    parser.add_argument("--page", type=Path, default=DEFAULT_PAGE)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--verify-text", type=str, default="Results for: Python",
        help='Text expected to appear after the instruction completes; page-specific default matches search_basic.html\'s JS behavior for the default instruction.',
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    summary = run_batch(
        page_path=args.page,
        instruction=args.instruction,
        runs=args.runs,
        verify_text=args.verify_text,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    _print_report(summary, args.report)


if __name__ == "__main__":
    main()
