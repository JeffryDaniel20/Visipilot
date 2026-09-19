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


# --- ordinal reference parsing (Phase C.2) ---------------------------------

def test_ordinal_word_extracted_and_stripped_from_phrase():
    steps = parse_instruction("Click the second search button.")
    assert steps[0].action == ActionKind.CLICK
    assert steps[0].target_phrase == "search button"
    assert steps[0].ordinal == 2


def test_open_is_a_click_synonym_with_ordinal():
    # The roadmap's own canonical example.
    steps = parse_instruction("Open the second result.")
    assert steps[0].action == ActionKind.CLICK
    assert steps[0].target_phrase == "result"
    assert steps[0].ordinal == 2


def test_first_and_third_ordinals():
    assert parse_instruction("Click the first result.")[0].ordinal == 1
    assert parse_instruction("Find the third item.")[0].ordinal == 3


def test_digit_ordinal_forms():
    steps = parse_instruction("Click the 3rd item.")
    assert steps[0].ordinal == 3
    assert steps[0].target_phrase == "item"


def test_last_ordinal_is_negative_one():
    steps = parse_instruction("Click the last result.")
    assert steps[0].ordinal == -1
    assert steps[0].target_phrase == "result"


def test_no_ordinal_word_leaves_ordinal_none():
    steps = parse_instruction("Click Subscribe.")
    assert steps[0].ordinal is None
    assert steps[0].target_phrase == "Subscribe"


def test_ordinal_word_with_nothing_after_it_is_not_stripped():
    # No noun left to match against if "first" were stripped -- treating
    # the whole clause as a literal (unmatchable) phrase is more honest
    # than guessing what was meant.
    steps = parse_instruction("Click the first.")
    assert steps[0].ordinal is None
    assert steps[0].target_phrase == "first"


def test_ordinal_does_not_affect_non_ordinal_steps_in_the_same_instruction():
    steps = parse_instruction("Find the search box, type Python, and click the second result.")
    assert steps[0].ordinal is None
    assert steps[1].ordinal is None
    assert steps[2].ordinal == 2
    assert steps[2].target_phrase == "result"
