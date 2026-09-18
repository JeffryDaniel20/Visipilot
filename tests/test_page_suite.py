"""Tests for visipilot.eval.page_suite."""
from __future__ import annotations

import pytest

from visipilot.eval.page_suite import PAGE_SUITE, DEFAULT_REPORT, run_suite


def test_page_suite_entries_reference_existing_files():
    assert len(PAGE_SUITE) >= 5, "Phase B expects meaningful test diversity, not a token single extra page"
    for spec in PAGE_SUITE:
        assert spec.page_path.exists(), f"{spec.name}: missing page file {spec.page_path}"
        assert spec.note, f"{spec.name}: every page must document what it stresses"
        assert "input" in spec.ground_truth_selectors
        assert "button" in spec.ground_truth_selectors


def test_page_suite_names_are_unique():
    names = [spec.name for spec in PAGE_SUITE]
    assert len(names) == len(set(names))


@pytest.mark.integration
def test_run_suite_real_small_scale(tmp_path):
    """Runs the actual suite end to end, real models, 1 run per page —
    proves the wiring (model reuse across pages, per-page ground truth,
    suite-level VRAM/RAM aggregation, JSON-serializable report) works.
    The real, meaningful-N (5 runs/page) evaluation is a separate,
    deliberate invocation — see implementation-plan.md Phase B — not
    something the regular test suite re-runs on every `pytest` call.
    """
    import json

    report_path = tmp_path / "page_suite_report.json"
    report = run_suite(runs_per_page=1, report_path=report_path)

    assert report_path.exists()
    assert set(report["pages"]) == {spec.name for spec in PAGE_SUITE}
    assert report["suite_peak_vram_mb"] > 0
    assert report["suite_peak_ram_mb"] > 0
    for name, summary in report["per_page"].items():
        assert summary["runs"] == 1, name

    # Must be exactly what main() writes to disk.
    json.dumps(report, default=str)
