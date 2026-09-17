"""Runs a parsed instruction (a sequence of FIND/TYPE/CLICK steps)
against a live page, resolving each step's target via the matcher and
executing it via the Action Executor.

Single pass, no retries: Instructions.md #7 requires bounded retries and
no unbounded action loops, and a real retry/re-perception policy is
Phase A.6/Phase C scope (state tracking across steps, clarification
paths). This runner satisfies "bounded" trivially — it stops at the
first step that can't be safely resolved — rather than half-implementing
retry logic ahead of the design work that should drive it.
"""
from __future__ import annotations

from playwright.sync_api import Page

from visipilot.action.executor import (
    AmbiguousTargetError,
    NoConfidentTargetError,
    click_element,
    resolve_single_candidate,
    type_into_element,
)
from visipilot.target_selection.instruction_parser import ActionKind, InstructionStep
from visipilot.target_selection.matcher import MatchCandidate, match_target
from visipilot.types import ActionOutcome, ActionRecord, SemanticUIState


def run_steps(page: Page, steps: list[InstructionStep], state: SemanticUIState) -> list[ActionRecord]:
    """Execute `steps` in order. Stops at the first step whose target
    can't be resolved safely (no candidates, low confidence, or a tie) —
    the returned list is shorter than `steps` in that case, and its last
    record's `outcome`/`error_message` says why.
    """
    meta = state.screenshot.meta
    records: list[ActionRecord] = []
    focused: MatchCandidate | None = None

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
            record = type_into_element(page, focused, meta, step.value or "")
            records.append(record)
            if record.outcome != ActionOutcome.SUCCESS:
                break

        elif step.action == ActionKind.CLICK:
            candidates = match_target(step.target_phrase, state)
            try:
                target = resolve_single_candidate(candidates)
            except (NoConfidentTargetError, AmbiguousTargetError) as exc:
                records.append(_failure_record("click", exc))
                break
            focused = target
            record = click_element(page, target, meta)
            records.append(record)
            if record.outcome != ActionOutcome.SUCCESS:
                break

    return records


def _failure_record(action: str, exc: Exception) -> ActionRecord:
    outcome = ActionOutcome.FAILED_AMBIGUOUS if isinstance(exc, AmbiguousTargetError) else ActionOutcome.FAILED_NO_TARGET
    return ActionRecord(action=action, outcome=outcome, error_message=str(exc))
