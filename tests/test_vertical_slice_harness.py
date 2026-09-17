"""Tests for visipilot.eval.vertical_slice — unit tests for the pure
summarization logic (synthetic run data, no browser/model needed), plus
a real, small-scale integration test of run_batch() itself.
"""
from __future__ import annotations

import pytest

from visipilot.eval.vertical_slice import (
    MAX_LATENCY_S,
    MAX_PEAK_RAM_MB,
    MAX_PEAK_VRAM_MB,
    MIN_CLICK_ACCURACY,
    MIN_RUNS_FOR_GATE,
    MIN_SUCCESS_RATE,
    _click_lands_in_bbox,
    _summarize,
)
from visipilot.types import BBox, CoordinateSpace


def make_run(success=True, click_checks=None, wall_time_s=2.0, failure_reason=None):
    return {
        "run_index": 0,
        "run_id": "abc123",
        "success": success,
        "failure_reason": failure_reason if not success else None,
        "wall_time_s": wall_time_s,
        "click_checks": click_checks or [],
        "click_accuracy_ok": all(click_checks) if click_checks else None,
        "action_outcomes": ["success"] if success else ["failed_ambiguous"],
        "verification_passed": success,
    }


# --- _click_lands_in_bbox ---------------------------------------------------

def test_click_lands_in_bbox_true():
    bbox = BBox(x=10, y=10, width=100, height=40, space=CoordinateSpace.VIEWPORT_CSS)
    assert _click_lands_in_bbox((50, 20), bbox) is True


def test_click_lands_in_bbox_false():
    bbox = BBox(x=10, y=10, width=100, height=40, space=CoordinateSpace.VIEWPORT_CSS)
    assert _click_lands_in_bbox((500, 20), bbox) is False


def test_click_lands_in_bbox_none_when_no_click_point():
    bbox = BBox(x=10, y=10, width=100, height=40, space=CoordinateSpace.VIEWPORT_CSS)
    assert _click_lands_in_bbox(None, bbox) is None


def test_click_lands_in_bbox_none_when_no_ground_truth():
    assert _click_lands_in_bbox((50, 20), None) is None


def test_click_lands_in_bbox_boundary_inclusive():
    bbox = BBox(x=0, y=0, width=100, height=40, space=CoordinateSpace.VIEWPORT_CSS)
    assert _click_lands_in_bbox((100, 40), bbox) is True  # x2/y2 edge


# --- _summarize: criteria pass/fail -----------------------------------------

def test_summarize_all_pass():
    runs = [make_run(click_checks=[True, True]) for _ in range(20)]
    summary = _summarize(runs, peak_vram_mb=1700.0, peak_ram_mb=3000.0)
    assert summary["success_rate"] == 1.0
    assert summary["click_accuracy"] == 1.0
    assert summary["criteria"]["success_rate_ok"] is True
    assert summary["criteria"]["click_accuracy_ok"] is True
    assert summary["criteria"]["vram_ok"] is True
    assert summary["criteria"]["ram_ok"] is True
    assert summary["overall_pass"] is True


def test_summarize_fails_below_min_runs():
    runs = [make_run(click_checks=[True]) for _ in range(5)]
    summary = _summarize(runs, peak_vram_mb=1700.0, peak_ram_mb=3000.0)
    assert summary["criteria"]["min_runs_met"] is False
    assert summary["overall_pass"] is False


def test_summarize_fails_on_low_success_rate():
    runs = [make_run(success=(i < 17), failure_reason="x", click_checks=[True]) for i in range(20)]
    summary = _summarize(runs, peak_vram_mb=1700.0, peak_ram_mb=3000.0)
    assert summary["success_rate"] == pytest.approx(17 / 20)
    assert summary["success_rate"] < MIN_SUCCESS_RATE
    assert summary["criteria"]["success_rate_ok"] is False
    assert summary["overall_pass"] is False
    assert len(summary["failures"]) == 3


def test_summarize_success_rate_at_exact_threshold_passes():
    # 18/20 = 0.90 exactly, and the criterion is >=.
    runs = [make_run(success=(i < 18), failure_reason="x", click_checks=[True]) for i in range(20)]
    summary = _summarize(runs, peak_vram_mb=1700.0, peak_ram_mb=3000.0)
    assert summary["success_rate"] == pytest.approx(MIN_SUCCESS_RATE)
    assert summary["criteria"]["success_rate_ok"] is True


def test_summarize_fails_on_low_click_accuracy():
    runs = [make_run(click_checks=[True, False]) for _ in range(20)]  # 50% accuracy
    summary = _summarize(runs, peak_vram_mb=1700.0, peak_ram_mb=3000.0)
    assert summary["click_accuracy"] < MIN_CLICK_ACCURACY
    assert summary["criteria"]["click_accuracy_ok"] is False
    assert summary["overall_pass"] is False


def test_summarize_no_checkable_clicks_is_not_a_silent_pass():
    runs = [make_run(click_checks=[]) for _ in range(20)]
    summary = _summarize(runs, peak_vram_mb=1700.0, peak_ram_mb=3000.0)
    assert summary["click_accuracy"] is None
    assert summary["criteria"]["click_accuracy_ok"] is False
    assert summary["overall_pass"] is False


def test_summarize_fails_on_latency():
    runs = [make_run(click_checks=[True], wall_time_s=6.0) for _ in range(20)]
    summary = _summarize(runs, peak_vram_mb=1700.0, peak_ram_mb=3000.0)
    assert summary["max_latency_s"] > MAX_LATENCY_S
    assert summary["criteria"]["latency_ok"] is False
    assert summary["overall_pass"] is False


def test_summarize_fails_on_vram():
    runs = [make_run(click_checks=[True]) for _ in range(20)]
    summary = _summarize(runs, peak_vram_mb=MAX_PEAK_VRAM_MB + 1, peak_ram_mb=3000.0)
    assert summary["criteria"]["vram_ok"] is False
    assert summary["overall_pass"] is False


def test_summarize_fails_on_ram():
    runs = [make_run(click_checks=[True]) for _ in range(20)]
    summary = _summarize(runs, peak_vram_mb=1700.0, peak_ram_mb=MAX_PEAK_RAM_MB + 1)
    assert summary["criteria"]["ram_ok"] is False
    assert summary["overall_pass"] is False


def test_summarize_vram_at_exact_threshold_passes():
    runs = [make_run(click_checks=[True]) for _ in range(20)]
    summary = _summarize(runs, peak_vram_mb=MAX_PEAK_VRAM_MB, peak_ram_mb=3000.0)
    assert summary["criteria"]["vram_ok"] is True


def test_summarize_empty_runs_does_not_crash():
    summary = _summarize([], peak_vram_mb=0.0, peak_ram_mb=0.0)
    assert summary["runs"] == 0
    assert summary["success_rate"] == 0.0
    assert summary["click_accuracy"] is None
    assert summary["overall_pass"] is False


# --- run_batch: real, small-scale integration test --------------------------

@pytest.mark.integration
def test_run_batch_real_small_scale():
    """Runs the actual harness end to end against the real page and real
    models, at a small scale (2 runs) for test speed — proves the whole
    wiring (server, browser reuse, page.goto reset between runs, DOM
    ground-truth scoring, VRAM/RAM measurement, JSON-serializable
    summary) works for real. The 20-run pass-gate evaluation itself is a
    separate, deliberate invocation (see implementation-plan.md), not
    something the test suite re-runs on every `pytest` call.
    """
    import json

    from visipilot.eval.vertical_slice import DEFAULT_PAGE, run_batch

    summary = run_batch(
        page_path=DEFAULT_PAGE,
        instruction="Find the search box, type Python, and click Search.",
        runs=2,
        verify_text="Results for: Python",
    )

    assert summary["runs"] == 2
    assert len(summary["run_results"]) == 2
    assert summary["peak_vram_mb"] > 0
    assert summary["peak_ram_mb"] > 0
    # Real evidence, not assumed: with the fill-color fix in place, both
    # runs are expected to succeed on this unmodified page.
    assert summary["successes"] == 2
    assert summary["click_accuracy"] == 1.0

    # The summary must be JSON-serializable exactly as main() writes it.
    json.dumps(summary, default=str)
