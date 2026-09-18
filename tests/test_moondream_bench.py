"""Unit tests for the pure, synthetic-input parts of
visipilot.eval.moondream_bench, plus config-integrity checks. Does not
load Moondream2 itself (network + GPU + a load that requires an
undocumented third-party-code monkeypatch to even run, and even then
produces unusable output — see implementation-plan.md B.9) — this only
locks in that the benchmark's own scoring logic and page config are
correct, so a result from a future, properly-loading revision would be
trustworthy.
"""
from __future__ import annotations

from pathlib import Path

from visipilot.eval.moondream_bench import ICON_PAGES, TESTPAGES_DIR, _matches_keywords


def test_matches_keywords_case_insensitive_substring():
    assert _matches_keywords("A Search Icon", ["search"]) is True


def test_matches_keywords_no_match():
    assert _matches_keywords("a blue circle", ["search", "magnify"]) is False


def test_matches_keywords_any_one_of_several_is_enough():
    assert _matches_keywords("looks like a gear or cog", ["setting", "gear", "cog"]) is True


def test_matches_keywords_garbage_output_does_not_spuriously_match():
    # Regression-style sanity check motivated by the real Moondream2
    # output found this milestone (long repetitive nonsense) -- garbage
    # text must not accidentally contain a keyword substring by chance
    # for this specific fixture set.
    garbage = "PhotPhotPhotKeKeKe Traditional Unterschiede difference between"
    assert _matches_keywords(garbage, ["search", "magnify", "find", "loupe", "glass"]) is False


def test_icon_pages_config_points_at_real_files():
    assert len(ICON_PAGES) == 4
    for name, page_file, sel, keywords in ICON_PAGES:
        assert (TESTPAGES_DIR / page_file).exists(), f"{page_file} missing on disk"
        assert sel.startswith("#")
        assert len(keywords) >= 3
