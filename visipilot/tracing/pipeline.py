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

from visipilot.action.clarification import Clarifier
from visipilot.action.runner import RetryPolicy, run_steps
from visipilot.action.verification import verify_text_present
from visipilot.capture.screenshot import capture_screenshot
from visipilot.perception.detector import UIDetector
from visipilot.perception.ocr import OCREngine
from visipilot.semantic.builder import build_semantic_state
from visipilot.target_selection.instruction_parser import parse_instruction
from visipilot.types import (
    ActionOutcome,
    ActionRecord,
    RetrySummary,
    RuntimeInfo,
    SemanticUIState,
    TraceRecord,
    VerificationResult,
)

logger = logging.getLogger("visipilot.pipeline")

DEFAULT_TRACE_DIR = Path(__file__).resolve().parent.parent.parent / "out" / "traces"


def _runtime_info() -> RuntimeInfo:
    return RuntimeInfo(
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def _verify_with_retry(
    ocr: OCREngine,
    page: Page,
    expected_text: str,
    reperceive,
    timings: dict[str, float],
    run_id: str,
) -> VerificationResult:
    """Checks `expected_text` against a fresh screenshot; on failure, if
    `reperceive` is given (budget allows one more re-perception this
    run), triggers exactly one more re-perception cycle and re-checks
    against ITS fresh screenshot before giving up — implementation-plan.md
    C.4, closing the "maximum 1 full re-perception cycle per step on
    verification failure" pass criterion the original plan specified but
    C.1 didn't yet implement (C.1 only covered pre-action staleness).

    Deliberately does NOT re-execute the action that was already taken:
    the click/type already happened and reported SUCCESS; if its visible
    effect just hasn't rendered yet (a slow search-results fade-in, a
    debounced update), re-perceiving gives it more real wall-clock time
    to appear before checking again. Re-clicking on a verification
    failure would risk a real, worse problem than a slow render — a
    second, unintended action (e.g. double-submitting a form) — which is
    exactly the kind of consequence Instructions.md #7's bounded-retry
    rule exists to prevent, not something a "helpful" retry should risk
    causing.
    """
    t0 = time.perf_counter()
    after_shot = capture_screenshot(page)
    verification = verify_text_present(ocr, after_shot.image_path, expected_text)
    timings["verification_ms"] = (time.perf_counter() - t0) * 1000
    if verification.passed or reperceive is None:
        return verification

    logger.info("run=%s stage=verification status=fail_retry", run_id)
    reperceive()  # discard the fresh SemanticUIState; only the fresh screenshot below is checked
    t1 = time.perf_counter()
    after_shot2 = capture_screenshot(page)
    retried_verification = verify_text_present(ocr, after_shot2.image_path, expected_text)
    timings["verification_retry_ms"] = (time.perf_counter() - t1) * 1000
    return retried_verification.model_copy(update={"retried": True})


def _summarize_retries(records: list[ActionRecord], verification: VerificationResult | None) -> RetrySummary:
    """Derived from the per-attempt records (and the verification result,
    if any) rather than counted separately, so the summary can never
    drift from what actually happened.
    """
    reperceptions = sum(1 for r in records if r.reperceived and r.attempt > 1)
    if verification is not None and verification.retried:
        reperceptions += 1
    return RetrySummary(
        total_attempts=len(records),
        retried_attempts=sum(1 for r in records if r.attempt > 1),
        reperceptions=reperceptions,
        clarifications_used=sum(1 for r in records if r.clarification_used),
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
    enable_retry: bool = True,
    clarifier: Clarifier | None = None,
    retry_policy: RetryPolicy | None = None,
) -> TraceRecord:
    """Run `instruction` end to end against `page`: screenshot ->
    detection -> OCR -> semantic state -> parse -> act (per step) ->
    optional pixels-first verification. Returns (and, by default, writes
    to `trace_dir/<run_id>.json`) a full TraceRecord.

    `verify_expected_text`, when given, is checked via a fresh
    post-action screenshot + OCR only if every action step succeeded —
    there's nothing meaningful to verify after a halted/failed run. A
    first failed check gets exactly one more re-perception cycle and one
    re-check before giving up (implementation-plan.md C.4), sharing the
    same `retry_policy` re-perception budget `enable_retry` governs — it
    does NOT re-click/re-type; see `_verify_with_retry`'s docstring for
    why re-executing the action on a verification failure is a real risk
    (e.g. double-submitting), not a safe default.

    `full_page`, when True, captures the entire scrollable page (not
    just the current viewport) for the initial perception pass — added
    for Phase B's below-the-fold test page (implementation-plan.md
    Phase B). This lets detection/OCR see off-screen content; it does
    NOT make the action executor scroll the viewport to reach it, so a
    grounded click point outside the current viewport still correctly
    fails via `is_within_viewport()` in visipilot/action/executor.py —
    scrolling the viewport to a resolved target is real, separate,
    not-yet-built functionality, not something this flag fakes.

    `enable_retry` (default True) gives the runner a re-perception
    callable, turning a detected stale screenshot into a bounded
    re-perceive-and-retry cycle instead of an immediate refusal
    (Instructions.md #7 / implementation-plan.md C.1). Set it False to
    get Phase B's detect-and-refuse behaviour — useful for measuring the
    two against each other, which is exactly how C.1's before/after
    numbers were produced. `clarifier` and `retry_policy` are passed
    straight through to `run_steps`; both default to the safe, unchanged
    behaviour (refuse on a tie; the standard bounded policy).

    The trace's `detected_elements`/`ocr_elements`/`fused_elements` and
    `screenshot_hash` always describe the *first* perception pass, the
    one the run's decisions started from; re-perception passes are
    accounted for in `retry_summary` and in each retried record's
    `reperceived` flag rather than silently overwriting the original
    evidence.
    """
    run_id = uuid.uuid4().hex[:12]
    timings: dict[str, float] = {}
    failure_reason: str | None = None
    resolved_policy = retry_policy or RetryPolicy()

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

    reperception_count = 0

    def _reperceive() -> SemanticUIState:
        # The same perception chain the run started with, re-run against
        # the page as it is *now*. Kept as a closure over the already-
        # loaded detector/OCR so a retry costs one more inference pass,
        # never a model reload.
        nonlocal reperception_count
        reperception_count += 1
        t_rp = time.perf_counter()
        fresh_shot = capture_screenshot(page, full_page=full_page)
        fresh_detected = detector.detect(fresh_shot.image_path)
        fresh_ocr = ocr.read(fresh_shot.image_path)
        fresh_state = build_semantic_state(fresh_shot, fresh_detected, fresh_ocr)
        elapsed = (time.perf_counter() - t_rp) * 1000
        timings[f"reperception_{reperception_count}_ms"] = elapsed
        logger.info(
            "run=%s stage=reperception status=ok pass=%d timing_ms=%.1f elements=%d",
            run_id, reperception_count, elapsed, len(fresh_state.elements),
        )
        return fresh_state

    t0 = time.perf_counter()
    action_records = run_steps(
        page,
        steps,
        state,
        reperceive=_reperceive if enable_retry else None,
        clarifier=clarifier,
        policy=retry_policy,
    )
    timings["action_ms"] = (time.perf_counter() - t0) * 1000
    if any(r.attempt > 1 or r.clarification_used for r in action_records):
        logger.info(
            "run=%s stage=action status=retries retried_attempts=%d clarifications=%d",
            run_id, sum(1 for r in action_records if r.attempt > 1),
            sum(1 for r in action_records if r.clarification_used),
        )

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
        verification = _verify_with_retry(
            ocr, page, verify_expected_text,
            reperceive=_reperceive if enable_retry and reperception_count < resolved_policy.max_reperceptions_per_run else None,
            timings=timings, run_id=run_id,
        )
        if verification.passed:
            logger.info(
                "run=%s stage=verification status=ok timing_ms=%.1f retried=%s",
                run_id, timings["verification_ms"], verification.retried,
            )
        else:
            failure_reason = f"verification failed: {verification.detail}"
            logger.warning(
                "run=%s stage=verification status=fail timing_ms=%.1f retried=%s reason=%r",
                run_id, timings["verification_ms"], verification.retried, failure_reason,
            )

    retry_summary = _summarize_retries(action_records, verification)

    record = TraceRecord(
        run_id=run_id,
        instruction=instruction,
        screenshot_hash=shot.image_hash,
        detected_elements=detected,
        ocr_elements=ocr_elements,
        fused_elements=state.elements,
        action_records=action_records,
        retry_summary=retry_summary,
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
