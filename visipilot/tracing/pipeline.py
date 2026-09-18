"""End-to-end traced pipeline: runs one instruction against a live page,
logging each stage (name, timing, pass/fail, failure reason) and writing
a full TraceRecord to disk. This is the first place every stage built so
far (capture -> detect -> OCR -> semantic build -> parse -> act ->
verify) is orchestrated behind one function, rather than only exercised
stage-by-stage in tests.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import torch
from playwright.sync_api import Page

from visipilot.action.runner import run_steps
from visipilot.action.verification import verify_text_present
from visipilot.capture.screenshot import capture_screenshot
from visipilot.perception.detector import UIDetector
from visipilot.perception.ocr import OCREngine
from visipilot.semantic.builder import build_semantic_state
from visipilot.target_selection.instruction_parser import parse_instruction
from visipilot.types import ActionOutcome, RuntimeInfo, TraceRecord, VerificationResult

logger = logging.getLogger("visipilot.pipeline")

DEFAULT_TRACE_DIR = Path(__file__).resolve().parent.parent.parent / "out" / "traces"


def _runtime_info() -> RuntimeInfo:
    return RuntimeInfo(
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def run_instruction(
    page: Page,
    instruction: str,
    detector: UIDetector,
    ocr: OCREngine,
    verify_expected_text: str | None = None,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    write_trace: bool = True,
    full_page: bool = False,
) -> TraceRecord:
    """Run `instruction` end to end against `page`: screenshot ->
    detection -> OCR -> semantic state -> parse -> act (per step) ->
    optional pixels-first verification. Returns (and, by default, writes
    to `trace_dir/<run_id>.json`) a full TraceRecord.

    `verify_expected_text`, when given, is checked via a fresh
    post-action screenshot + OCR only if every action step succeeded —
    there's nothing meaningful to verify after a halted/failed run.

    `full_page`, when True, captures the entire scrollable page (not
    just the current viewport) for the initial perception pass — added
    for Phase B's below-the-fold test page (implementation-plan.md
    Phase B). This lets detection/OCR see off-screen content; it does
    NOT make the action executor scroll the viewport to reach it, so a
    grounded click point outside the current viewport still correctly
    fails via `is_within_viewport()` in visipilot/action/executor.py —
    scrolling the viewport to a resolved target is real, separate,
    not-yet-built functionality, not something this flag fakes.
    """
    run_id = uuid.uuid4().hex[:12]
    timings: dict[str, float] = {}
    failure_reason: str | None = None

    logger.info("run=%s stage=screenshot status=start", run_id)
    t0 = time.perf_counter()
    shot = capture_screenshot(page, full_page=full_page)
    timings["screenshot_ms"] = (time.perf_counter() - t0) * 1000
    logger.info("run=%s stage=screenshot status=ok timing_ms=%.1f", run_id, timings["screenshot_ms"])

    t0 = time.perf_counter()
    detected = detector.detect(shot.image_path)
    timings["detection_ms"] = (time.perf_counter() - t0) * 1000
    logger.info(
        "run=%s stage=detection status=ok timing_ms=%.1f elements=%d",
        run_id, timings["detection_ms"], len(detected),
    )

    t0 = time.perf_counter()
    ocr_elements = ocr.read(shot.image_path)
    timings["ocr_ms"] = (time.perf_counter() - t0) * 1000
    logger.info(
        "run=%s stage=ocr status=ok timing_ms=%.1f elements=%d",
        run_id, timings["ocr_ms"], len(ocr_elements),
    )

    t0 = time.perf_counter()
    state = build_semantic_state(shot, detected, ocr_elements)
    timings["build_semantic_state_ms"] = (time.perf_counter() - t0) * 1000
    logger.info("run=%s stage=build_semantic_state status=ok timing_ms=%.1f", run_id, timings["build_semantic_state_ms"])

    t0 = time.perf_counter()
    steps = parse_instruction(instruction)
    timings["parse_instruction_ms"] = (time.perf_counter() - t0) * 1000
    logger.info("run=%s stage=parse_instruction status=ok timing_ms=%.1f steps=%d", run_id, timings["parse_instruction_ms"], len(steps))

    t0 = time.perf_counter()
    action_records = run_steps(page, steps, state)
    timings["action_ms"] = (time.perf_counter() - t0) * 1000

    if not action_records:
        failure_reason = "no steps executed (instruction did not parse into any recognized step)"
        logger.warning("run=%s stage=action status=fail reason=%r", run_id, failure_reason)
    elif action_records[-1].outcome != ActionOutcome.SUCCESS:
        last = action_records[-1]
        failure_reason = f"{last.action} failed: {last.outcome.value} ({last.error_message or 'no detail'})"
        logger.warning("run=%s stage=action status=fail timing_ms=%.1f reason=%r", run_id, timings["action_ms"], failure_reason)
    else:
        logger.info("run=%s stage=action status=ok timing_ms=%.1f steps_completed=%d", run_id, timings["action_ms"], len(action_records))

    verification: VerificationResult | None = None
    if verify_expected_text is not None and failure_reason is None:
        t0 = time.perf_counter()
        after_shot = capture_screenshot(page)
        verification = verify_text_present(ocr, after_shot.image_path, verify_expected_text)
        timings["verification_ms"] = (time.perf_counter() - t0) * 1000
        if verification.passed:
            logger.info("run=%s stage=verification status=ok timing_ms=%.1f", run_id, timings["verification_ms"])
        else:
            failure_reason = f"verification failed: {verification.detail}"
            logger.warning("run=%s stage=verification status=fail timing_ms=%.1f reason=%r", run_id, timings["verification_ms"], failure_reason)

    record = TraceRecord(
        run_id=run_id,
        instruction=instruction,
        screenshot_hash=shot.image_hash,
        detected_elements=detected,
        ocr_elements=ocr_elements,
        fused_elements=state.elements,
        action_records=action_records,
        verification=verification,
        stage_timings_ms=timings,
        model_versions={"detector": detector.model_id, "ocr": "easyocr"},
        runtime_info=_runtime_info(),
        failure_reason=failure_reason,
    )

    if write_trace:
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{run_id}.json"
        trace_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        logger.info("run=%s trace_written=%s", run_id, trace_path)

    return record
