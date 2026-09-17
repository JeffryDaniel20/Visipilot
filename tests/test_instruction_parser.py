"""Unit tests for visipilot.target_selection.instruction_parser."""
from __future__ import annotations

from visipilot.target_selection.instruction_parser import ActionKind, parse_instruction


def test_vertical_slice_instruction():
    steps = parse_instruction("Find the search box, type Python, and click Search.")
    assert len(steps) == 3
    assert steps[0].action == ActionKind.FIND
    assert steps[0].target_phrase == "search box"
    assert steps[1].action == ActionKind.TYPE
    assert steps[1].value == "Python"
    assert steps[2].action == ActionKind.CLICK
    assert steps[2].target_phrase == "Search"


def test_single_click_clause():
    steps = parse_instruction("Click Subscribe.")
    assert len(steps) == 1
    assert steps[0].action == ActionKind.CLICK
    assert steps[0].target_phrase == "Subscribe"


def test_click_on_variant():
    steps = parse_instruction("Click on the Subscribe button.")
    assert len(steps) == 1
    assert steps[0].action == ActionKind.CLICK
    assert steps[0].target_phrase == "Subscribe button"


def test_type_with_into_suffix_stripped():
    steps = parse_instruction("Type hello into the search box.")
    assert len(steps) == 1
    assert steps[0].action == ActionKind.TYPE
    assert steps[0].value == "hello"


def test_locate_synonym_for_find():
    steps = parse_instruction("Locate the newsletter field.")
    assert steps[0].action == ActionKind.FIND
    assert steps[0].target_phrase == "newsletter field"


def test_unparseable_clause_silently_dropped():
    steps = parse_instruction("Please do something vague here.")
    assert steps == []


def test_empty_instruction():
    assert parse_instruction("") == []
    assert parse_instruction("   ") == []
