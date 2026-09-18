"""Runs a parsed instruction (a sequence of FIND/TYPE/CLICK steps)
against a live page, resolving each step's target via the matcher and
executing it via the Action Executor.

Single pass, no retries: Instructions.md #7 requires bounded retries and
no unbounded action loops, and a real retry/re-perception policy is
Phase A.6/Phase C scope (state tracking across steps, clarification
paths). This runner satisfies "bounded" trivially — it stops at the
first step that can't be safely resolved — rather than half-implementing
retry logic ahead of the design work that should drive it.

Before every CLICK/TYPE action (the only steps that act on coordinates
computed from `state.screenshot`), a cheap re-screenshot-and-compare
checks the live page still matches what perception saw — Instructions.md
#7's stale-screenshot rule, implemented as detect-and-refuse rather than
a full automatic re-perception-and-retry loop: the latter is real,
separate multi-step/retry-policy design (state tracking across a retry,
how many attempts, what changes on a retry) that belongs with Phase C's
clarification/retry work, not bolted on here ahead of that design — the
same scoping call A.5 already made for CLICK/TYPE's own retry policy.

The comparison reference is rolled forward after every successful
action, not pinned to the original `state.screenshot` for the whole
run — found necessary by testing against a real multi-step instruction,
not assumed: a TYPE step's own echoed input text legitimately changes
the page's rendered pixels, so comparing a later CLICK step against the
*pre-typing* screenshot always looks "stale" even when nothing external
happened. Comparing each step against the page state left by the
previous step correctly catches only genuinely external changes (a late
banner, a layout shift) between one action and the next.
"""
from __future__ import annotations

from typing import Callable

from playwright.sync_api import Page

from visipilot.action.executor import (
    AmbiguousTargetError,
    NoConfidentTargetError,
    click_element,
    resolve_single_candidate,
    type_into_element,
)
from visipilot.capture.screenshot import capture_screenshot, screenshot_has_changed
from visipilot.target_selection.instruction_parser import ActionKind, InstructionStep
from visipilot.target_selection.matcher import MatchCandidate, match_target
from visipilot.types import ActionOutcome, ActionRecord, Screenshot, SemanticUIState

StaleCheck = Callable[[Page, Screenshot], bool]


def run_steps(
    page: Page,
    steps: list[InstructionStep],
    state: SemanticUIState,
    stale_check: StaleCheck | None = screenshot_has_changed,
) -> list[ActionRecord]:
    """Execute `steps` in order. Stops at the first step whose target
    can't be resolved safely (no candidates, low confidence, or a tie),
    or whose target coordinates are no longer trustworthy because the
    live page has changed since the last known-good screenshot — the
    returned list is shorter than `steps` in that case, and its last
    record's `outcome`/`error_message` says why.

    `stale_check` defaults to a real, live re-screenshot comparison —
    pass `None` to disable it (e.g. in unit tests using a mocked
    `Page` that can't produce a real screenshot) or a stub for
    deterministic tests of the refusal path itself.
    """
    meta = state.screenshot.meta
    records: list[ActionRecord] = []
    focused: MatchCandidate | None = None
    reference_screenshot = state.screenshot

    for step in steps:
        if step.action == ActionKind.FIND:
            candidates = match_target(step.target_phrase, state)
            try:
                focused = resolve_single_candidate(candidates)
            except (NoConfidentTargetError, AmbiguousTargetError) as exc:
                records.append(_failure_record("find", exc))
                break
            records.append(ActionRecord(action="find", target_element_id=focused.element.id, outcome=ActionOutcome.SUCCESS))

        elif step.action == ActionKind.TYPE:
            if focused is None:
                records.append(
                    ActionRecord(
                        action="type",
                        value=step.value,
                        outcome=ActionOutcome.FAILED_NO_TARGET,
                        error_message="no prior resolved element to type into",
                    )
                )
                break
            if stale_check is not None and stale_check(page, reference_screenshot):
                records.append(_stale_record("type", focused, step.value))
                break
            record = type_into_element(page, focused, meta, step.value or "")
            records.append(record)
            if record.outcome != ActionOutcome.SUCCESS:
                break
            if stale_check is not None:
                reference_screenshot = capture_screenshot(page, full_page=reference_screenshot.meta.full_page)

        elif step.action == ActionKind.CLICK:
            candidates = match_target(step.target_phrase, state)
            try:
                target = resolve_single_candidate(candidates)
            except (NoConfidentTargetError, AmbiguousTargetError) as exc:
                records.append(_failure_record("click", exc))
                break
            focused = target
            if stale_check is not None and stale_check(page, reference_screenshot):
                records.append(_stale_record("click", target))
                break
            record = click_element(page, target, meta)
            records.append(record)
            if record.outcome != ActionOutcome.SUCCESS:
                break
            if stale_check is not None:
                reference_screenshot = capture_screenshot(page, full_page=reference_screenshot.meta.full_page)

    return records


def _failure_record(action: str, exc: Exception) -> ActionRecord:
    outcome = ActionOutcome.FAILED_AMBIGUOUS if isinstance(exc, AmbiguousTargetError) else ActionOutcome.FAILED_NO_TARGET
    return ActionRecord(action=action, outcome=outcome, error_message=str(exc))


def _stale_record(action: str, candidate: MatchCandidate, value: str | None = None) -> ActionRecord:
    return ActionRecord(
        action=action,
        target_element_id=candidate.element.id,
        value=value,
        outcome=ActionOutcome.FAILED_STALE_SCREENSHOT,
        error_message="page content changed since the perception screenshot was captured; refusing to act on now-unverified coordinates",
    )
