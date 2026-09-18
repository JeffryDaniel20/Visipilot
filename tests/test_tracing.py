"""Tests for visipilot.tracing.pipeline — real end-to-end runs against
the real controlled test page, real models. Verifies the TraceRecord is
correct, written to disk, and that structured logging covers every stage
(Instructions.md A.6: "stage name, timing, pass/fail, failure reason").
"""
from __future__ import annotations

import json
import logging

import pytest

from visipilot.capture.screenshot import launch_page
from visipilot.testserver import TestPageServer
from visipilot.tracing.pipeline import run_instruction
from visipilot.types import ActionOutcome

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def test_server():
    with TestPageServer(port=0) as server:
        yield server


@pytest.fixture(scope="module")
def models():
    from visipilot.perception.detector import UIDetector
    from visipilot.perception.ocr import OCREngine

    return UIDetector(), OCREngine(gpu=True)


def test_successful_run_produces_complete_trace_record(test_server, models, tmp_path, caplog):
    detector, ocr = models
    url = test_server.url("search_basic.html")

    with caplog.at_level(logging.INFO, logger="visipilot.pipeline"):
        with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
            record = run_instruction(
                page,
                "Click Subscribe.",
                detector,
                ocr,
                trace_dir=tmp_path,
            )

    assert record.failure_reason is None
    assert len(record.action_records) == 1
    assert record.action_records[0].outcome == ActionOutcome.SUCCESS
    assert record.detected_elements
    assert record.ocr_elements
    assert record.fused_elements
    assert record.runtime_info.device in ("cuda", "cpu")
    assert record.model_versions["ocr"] == "easyocr"
    assert set(record.stage_timings_ms) >= {
        "screenshot_ms", "detection_ms", "ocr_ms", "build_semantic_state_ms", "parse_instruction_ms", "action_ms",
    }
    assert all(v >= 0 for v in record.stage_timings_ms.values())

    # every stage logged with a timing and an ok/fail status
    messages = [r.message for r in caplog.records]
    for stage in ["screenshot", "detection", "ocr", "build_semantic_state", "parse_instruction", "action"]:
        assert any(f"stage={stage}" in m for m in messages), f"no log line for stage={stage}"


def test_trace_written_to_disk_and_round_trips(test_server, models, tmp_path):
    detector, ocr = models
    url = test_server.url("search_basic.html")

    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        record = run_instruction(page, "Click Subscribe.", detector, ocr, trace_dir=tmp_path)

    trace_path = tmp_path / f"{record.run_id}.json"
    assert trace_path.exists()

    from visipilot.types import TraceRecord

    raw = json.loads(trace_path.read_text(encoding="utf-8"))
    reloaded = TraceRecord.model_validate(raw)
    assert reloaded.run_id == record.run_id
    assert reloaded.action_records[0].outcome == ActionOutcome.SUCCESS


def test_full_vertical_slice_instruction_traced_end_to_end(test_server, models, tmp_path):
    """The instruction now genuinely succeeds end to end (see the fill-
    color matcher fix in implementation-plan.md A.6) -- traced here with
    verification against the real "Results for: Python" text.
    """
    detector, ocr = models
    url = test_server.url("search_basic.html")

    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        record = run_instruction(
            page,
            "Find the search box, type Python, and click Search.",
            detector,
            ocr,
            verify_expected_text="Results for: Python",
            trace_dir=tmp_path,
        )

    assert record.failure_reason is None
    assert [a.action for a in record.action_records] == ["find", "type", "click"]
    assert all(a.outcome == ActionOutcome.SUCCESS for a in record.action_records)
    assert record.verification is not None
    assert record.verification.passed is True
    assert "verification_ms" in record.stage_timings_ms


def test_failed_run_records_failure_reason_and_still_writes_trace(test_server, models, tmp_path):
    """A CLICK target that matches nothing must be traced as a clean
    failure, not an exception, and the trace must still be written —
    failed runs are exactly what you need the trace for.
    """
    detector, ocr = models
    url = test_server.url("search_basic.html")

    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        record = run_instruction(page, "Click NonexistentThingXYZ.", detector, ocr, trace_dir=tmp_path)

    assert record.failure_reason is not None
    assert (tmp_path / f"{record.run_id}.json").exists()


# --- Phase C.1: bounded re-perception retry, against a really-mutating page ---

VERTICAL_SLICE = "Find the search box, type Python, and click Search."


def test_dynamic_page_refuses_without_retry_but_succeeds_with_it(test_server, models, tmp_path):
    """The milestone's headline before/after, measured in one test so the
    two behaviours can't drift apart: on `search_dynamic.html` (a banner
    appears ~200ms after load, shifting the search row), Phase B's
    detect-and-refuse safely halts, and Phase C's bounded re-perception
    retry completes the same instruction correctly.
    """
    detector, ocr = models
    url = test_server.url("search_dynamic.html")

    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        without_retry = run_instruction(
            page, VERTICAL_SLICE, detector, ocr,
            verify_expected_text="Results for: Python", trace_dir=tmp_path, enable_retry=False,
        )

    assert without_retry.failure_reason is not None
    assert without_retry.action_records[-1].outcome == ActionOutcome.FAILED_STALE_SCREENSHOT
    assert without_retry.retry_summary.reperceptions == 0

    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        with_retry = run_instruction(
            page, VERTICAL_SLICE, detector, ocr,
            verify_expected_text="Results for: Python", trace_dir=tmp_path, enable_retry=True,
        )

    assert with_retry.failure_reason is None, with_retry.failure_reason
    assert with_retry.verification is not None and with_retry.verification.passed
    # The retry actually happened and is visible, rather than the page
    # having quietly settled on its own.
    assert with_retry.retry_summary.reperceptions >= 1
    assert with_retry.retry_summary.retried_attempts >= 1
    retried = [r for r in with_retry.action_records if r.attempt > 1]
    assert retried and all(r.reperceived for r in retried)
    assert any(r.outcome == ActionOutcome.FAILED_STALE_SCREENSHOT for r in with_retry.action_records)
    assert with_retry.action_records[-1].outcome == ActionOutcome.SUCCESS


def test_static_page_costs_no_retries_with_retry_enabled(test_server, models, tmp_path):
    # Regression guard: enabling retry must not add perception passes on
    # a page that never changes.
    detector, ocr = models
    url = test_server.url("search_basic.html")

    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        record = run_instruction(
            page, VERTICAL_SLICE, detector, ocr,
            verify_expected_text="Results for: Python", trace_dir=tmp_path, enable_retry=True,
        )

    assert record.failure_reason is None
    assert record.retry_summary.reperceptions == 0
    assert record.retry_summary.retried_attempts == 0
    assert not any(k.startswith("reperception_") for k in record.stage_timings_ms)


def test_retry_summary_and_records_round_trip_through_the_trace_file(test_server, models, tmp_path):
    # Retries must survive into the on-disk artifact evaluation reads,
    # not just live in memory.
    detector, ocr = models
    url = test_server.url("search_dynamic.html")

    with launch_page(url, viewport_width=1280, viewport_height=800, device_scale_factor=1.0) as page:
        record = run_instruction(
            page, VERTICAL_SLICE, detector, ocr, trace_dir=tmp_path, enable_retry=True,
        )

    written = json.loads((tmp_path / f"{record.run_id}.json").read_text(encoding="utf-8"))
    assert written["retry_summary"]["reperceptions"] == record.retry_summary.reperceptions
    assert [r["attempt"] for r in written["action_records"]] == [r.attempt for r in record.action_records]
    assert [r["reperceived"] for r in written["action_records"]] == [r.reperceived for r in record.action_records]
