"""Tests for visipilot.eval.latency_profile (Phase D.1)."""
from __future__ import annotations

import pytest

from visipilot.eval.latency_profile import _bucket_stage_key, _stats, run_profile


def test_bucket_stage_key_groups_variable_named_reperception_keys():
    assert _bucket_stage_key("reperception_1_ms") == "reperception_ms"
    assert _bucket_stage_key("reperception_2_ms") == "reperception_ms"


def test_bucket_stage_key_leaves_fixed_keys_unchanged():
    assert _bucket_stage_key("screenshot_ms") == "screenshot_ms"
    assert _bucket_stage_key("verification_retry_ms") == "verification_retry_ms"


def test_stats_computes_expected_aggregates():
    s = _stats([100.0, 200.0, 300.0])
    assert s["count"] == 3
    assert s["total_ms"] == 600.0
    assert s["mean_ms"] == 200.0
    assert s["median_ms"] == 200.0
    assert s["min_ms"] == 100.0
    assert s["max_ms"] == 300.0
    assert s["stdev_ms"] > 0


def test_stats_single_value_has_zero_stdev():
    s = _stats([42.0])
    assert s["stdev_ms"] == 0.0
    assert s["mean_ms"] == 42.0


@pytest.mark.integration
def test_run_profile_real_small_scale(tmp_path):
    """Runs the real profiler end to end, real models, 1 run per page --
    proves the wiring (trace collection, stage bucketing, percentage
    normalization, isolated match_target timing, VRAM/RAM aggregation)
    works. The real, meaningful-N (5 runs/page) profiling run is a
    separate, deliberate invocation -- see implementation-plan.md D.1 --
    not something the regular test suite re-runs on every `pytest` call.
    """
    report_path = tmp_path / "latency_profile_report.json"
    report = run_profile(runs_per_page=1, report_path=report_path)

    assert report_path.exists()
    assert report["total_runs"] == 9  # one run per page in PAGE_SUITE
    assert report["peak_vram_mb"] > 0
    assert report["peak_ram_mb"] > 0

    # Every real stage that always runs must be present and consistent.
    for stage in ("screenshot_ms", "detection_ms", "ocr_ms", "action_ms"):
        assert stage in report["per_stage"]
        assert report["per_stage"][stage]["count"] == 9

    # Percentages must be real fractions of a real total, not placeholders.
    total_pct = sum(s["pct_of_total"] for s in report["per_stage"].values())
    assert 99.0 <= total_pct <= 101.0  # rounding tolerance

    assert report["stage_ranking_by_total_time"][0] in report["per_stage"]
    assert report["isolated_match_target_mean_ms"] is not None
    assert report["isolated_match_target_mean_ms"] >= 0
