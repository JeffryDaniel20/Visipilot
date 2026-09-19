"""Phase D.1 — measurement-first latency/resource profiling of the real
end-to-end pipeline: where is time and VRAM/RAM actually spent, across
screenshot capture, OWLv2 detection, OCR, fusion/semantic-state build,
target selection, action execution, pixels-first verification, and any
retry/re-perception paths (Phase C.1/C.5)?

Deliberately measurement-only — no runtime code is changed here, and no
optimization (quantization, ONNX, TensorRT) is attempted before this
milestone's own numbers say where the real cost is, per
implementation-plan.md Phase D's explicit "measurement-first" framing.

Reuses `visipilot.eval.page_suite.PAGE_SUITE` (the real 9-page Phase B
fixture set, real instructions, real verify-text) and the real
`run_instruction()` pipeline exactly as production/every other
evaluation module calls it — not a synthetic or mocked stand-in. Every
`TraceRecord.stage_timings_ms` already recorded by the pipeline
(screenshot_ms, detection_ms, ocr_ms, build_semantic_state_ms,
parse_instruction_ms, action_ms, verification_ms, reperception_N_ms,
verification_retry_ms) is aggregated from real, on-disk trace files
written during this run — not re-derived or estimated.

One real gap in that existing instrumentation: `action_ms` wraps the
whole `run_steps()` call, which includes BOTH target-selection
(`match_target`) and actual action execution (`click_element`/
`type_into_element`) — the trace does not separate them. Rather than
add new runtime instrumentation ahead of knowing whether it would even
matter (this milestone's whole point), this module also runs one
direct, isolated measurement of `match_target()` alone, against the
same real per-page `SemanticUIState` the traced runs already built,
reported alongside (never double-counted into) the aggregate.

Usage:
    python -m visipilot.eval.latency_profile --runs-per-page 5 --report out/latency_profile_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import torch

from visipilot.capture.screenshot import capture_screenshot, launch_page
from visipilot.eval.page_suite import PAGE_SUITE, REPO_ROOT, TESTPAGES_DIR
from visipilot.eval.vertical_slice import run_batch
from visipilot.perception.detector import UIDetector
from visipilot.perception.ocr import OCREngine
from visipilot.semantic.builder import build_semantic_state
from visipilot.target_selection.instruction_parser import parse_instruction
from visipilot.target_selection.matcher import match_target
from visipilot.testserver import TestPageServer

logger = logging.getLogger("visipilot.eval.latency_profile")

DEFAULT_REPORT = REPO_ROOT / "out" / "latency_profile_report.json"
DEFAULT_TRACE_DIR = REPO_ROOT / "out" / "traces_latency_profile"
DEFAULT_RUNS_PER_PAGE = 5  # matches page_suite's own sample size, same rationale
MATCH_TARGET_ISOLATED_CALLS = 50  # per page, for a stable mean on a sub-ms operation

# Every stage key that can appear in a real TraceRecord.stage_timings_ms.
# reperception_N_ms/verification_retry_ms are variable-named per run, so
# they're bucketed under these two group labels instead.
_FIXED_STAGES = [
    "screenshot_ms", "detection_ms", "ocr_ms",
    "build_semantic_state_ms", "parse_instruction_ms",
    "action_ms", "verification_ms",
]


def _bucket_stage_key(key: str) -> str:
    if key.startswith("reperception_"):
        return "reperception_ms"
    if key == "verification_retry_ms":
        return "verification_retry_ms"
    return key


def _collect_trace_timings(trace_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))["stage_timings_ms"] for p in trace_dir.glob("*.json")]


def _stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "total_ms": round(sum(values), 1),
        "mean_ms": round(statistics.mean(values), 2),
        "median_ms": round(statistics.median(values), 2),
        "stdev_ms": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }


def _measure_isolated_match_target(detector: UIDetector, ocr: OCREngine, spec) -> dict | None:
    """Times `match_target()` alone, against a real SemanticUIState built
    from this page (a fresh detect+OCR+build cycle — not reused from the
    traced runs, so this measurement never touches/affects the wall-clock
    numbers being aggregated from those traces).
    """
    steps = parse_instruction(spec.instruction)
    target_steps = [s for s in steps if s.target_phrase]
    if not target_steps:
        return None

    with TestPageServer(directory=str(TESTPAGES_DIR), port=0) as server:
        url = server.url(spec.page_path.name)
        with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
            shot = capture_screenshot(page, full_page=spec.full_page)
            detected = detector.detect(shot.image_path)
            ocr_elements = ocr.read(shot.image_path)
            state = build_semantic_state(shot, detected, ocr_elements)

    phrase = target_steps[0].target_phrase
    durations = []
    for _ in range(MATCH_TARGET_ISOLATED_CALLS):
        t0 = time.perf_counter()
        match_target(phrase, state)
        durations.append((time.perf_counter() - t0) * 1000)
    return _stats(durations)


def run_profile(runs_per_page: int = DEFAULT_RUNS_PER_PAGE, report_path: Path = DEFAULT_REPORT) -> dict:
    trace_dir = DEFAULT_TRACE_DIR
    if trace_dir.exists():
        for f in trace_dir.glob("*.json"):
            f.unlink()  # start clean so this run's aggregate isn't polluted by a prior profiling session
    trace_dir.mkdir(parents=True, exist_ok=True)

    detector = UIDetector()
    ocr = OCREngine(gpu=True)
    torch.cuda.reset_peak_memory_stats()

    per_page_isolated_match: dict[str, dict] = {}
    peak_ram_mb = 0.0

    for spec in PAGE_SUITE:
        logger.info("=== profiling page=%s (%d runs) ===", spec.name, runs_per_page)
        summary = run_batch(
            page_path=spec.page_path,
            instruction=spec.instruction,
            runs=runs_per_page,
            verify_text=spec.verify_text,
            ground_truth_selectors=spec.ground_truth_selectors,
            detector=detector,
            ocr=ocr,
            full_page=spec.full_page,
            trace_dir=trace_dir,
            min_runs_for_gate=runs_per_page,
        )
        peak_ram_mb = max(peak_ram_mb, summary["peak_ram_mb"])
        isolated = _measure_isolated_match_target(detector, ocr, spec)
        if isolated is not None:
            per_page_isolated_match[spec.name] = isolated

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    all_timings = _collect_trace_timings(trace_dir)
    per_stage_values: dict[str, list[float]] = {}
    for timings in all_timings:
        for key, ms in timings.items():
            bucket = _bucket_stage_key(key)
            per_stage_values.setdefault(bucket, []).append(ms)

    per_stage_stats = {stage: _stats(vals) for stage, vals in per_stage_values.items()}
    total_traced_ms = sum(s["total_ms"] for s in per_stage_stats.values())
    for stage, s in per_stage_stats.items():
        s["pct_of_total"] = round(100 * s["total_ms"] / total_traced_ms, 1) if total_traced_ms else 0.0

    ranked = sorted(per_stage_stats.items(), key=lambda kv: kv[1]["total_ms"], reverse=True)

    all_match_target_means = [s["mean_ms"] for s in per_page_isolated_match.values()]

    report = {
        "runs_per_page": runs_per_page,
        "pages_profiled": [spec.name for spec in PAGE_SUITE],
        "total_runs": len(all_timings),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "peak_ram_mb": round(peak_ram_mb, 1),
        "per_stage": {stage: stats for stage, stats in ranked},
        "stage_ranking_by_total_time": [stage for stage, _ in ranked],
        "isolated_match_target_ms_per_page": per_page_isolated_match,
        "isolated_match_target_mean_ms": round(statistics.mean(all_match_target_means), 3) if all_match_target_means else None,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _print_report(report: dict, report_path: Path) -> None:
    print(f"\n=== Latency profile: {report['total_runs']} real runs across {len(report['pages_profiled'])} pages ===")
    print(f"{'stage':28s} {'total_ms':>10s} {'%':>6s} {'mean_ms':>9s} {'median_ms':>10s} {'stdev_ms':>9s} {'n':>5s}")
    for stage, s in report["per_stage"].items():
        print(f"{stage:28s} {s['total_ms']:10.1f} {s['pct_of_total']:5.1f}% {s['mean_ms']:9.2f} {s['median_ms']:10.2f} {s['stdev_ms']:9.2f} {s['count']:5d}")
    print(f"\nDominant stage by total wall-clock time: {report['stage_ranking_by_total_time'][0]}")
    print(f"Isolated match_target() mean across pages: {report['isolated_match_target_mean_ms']} ms/call "
          f"(already included inside action_ms above -- reported separately, not double-counted)")
    print(f"\nPeak VRAM this profiling session: {report['peak_vram_mb']:.1f} MB")
    print(f"Peak RAM this profiling session: {report['peak_ram_mb']:.1f} MB")
    print(f"Report written to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VisiPilot Phase D.1 latency/resource profiling")
    parser.add_argument("--runs-per-page", type=int, default=DEFAULT_RUNS_PER_PAGE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = run_profile(runs_per_page=args.runs_per_page, report_path=args.report)
    _print_report(report, args.report)


if __name__ == "__main__":
    main()
