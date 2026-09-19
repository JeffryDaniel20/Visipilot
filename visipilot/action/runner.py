"""Runs a parsed instruction (a sequence of FIND/TYPE/CLICK steps)
against a live page, resolving each step's target via the matcher and
executing it via the Action Executor.

**Bounded retry with re-perception (Phase C).** Before every CLICK/TYPE
action (the only steps that act on coordinates computed from
`state.screenshot`), a cheap re-screenshot-and-compare checks the live
page still matches what perception saw — Instructions.md #7's
stale-screenshot rule. Phase B implemented that as detect-and-refuse;
Phase C turns it into the controlled re-perception cycle the same rule
actually asks for: on a detected change, the runner re-runs perception
(via the injected `reperceive` callable), re-resolves the step's target
against the *fresh* state, and retries the action — up to a hard,
explicit limit.

Every limit is explicit and finite (`RetryPolicy`), and every attempt —
including the failed ones that triggered a retry — is recorded as its
own `ActionRecord` carrying `attempt`/`reperceived`, so a retry can
never happen invisibly. There is no unbounded loop anywhere: each step
gets at most `max_attempts_per_step` attempts, and the whole run shares
a single `max_reperceptions_per_run` budget, so N steps cannot multiply
into N x M perception cycles. Exhausting either limit is a distinct,
terminal `FAILED_RETRY_EXHAUSTED` outcome, not a silent give-up.

**Ambiguity is deliberately not retried.** Re-perceiving an unchanged
page yields the same tied candidates, so retrying it would be a
guaranteed-useless loop. Ambiguity instead goes to the optional injected
`clarifier` (visipilot/action/clarification.py); with no clarifier wired
up — the default — behaviour is exactly the pre-Phase-C safe refusal.

The staleness comparison reference is rolled forward after every
successful action, not pinned to the original `state.screenshot` for the
whole run — found necessary by testing against a real multi-step
instruction, not assumed: a TYPE step's own echoed input text
legitimately changes the page's rendered pixels, so comparing a later
CLICK step against the *pre-typing* screenshot always looks "stale" even
when nothing external happened. Comparing each step against the page
state left by the previous step correctly catches only genuinely
external changes (a late banner, a layout shift) between one action and
the next.
"""
from __future__ import annotations

import logging
from typing import Callable

from playwright.sync_api import Page
from pydantic import BaseModel

from visipilot.action.clarification import ClarificationRequest, Clarifier
from visipilot.action.executor import (
    AmbiguousTargetError,
    NoConfidentTargetError,
    OrdinalOutOfRangeError,
    click_element,
    resolve_single_candidate,
    type_into_element,
)
from visipilot.capture.screenshot import capture_screenshot, screenshot_has_changed
from visipilot.target_selection.instruction_parser import ActionKind, InstructionStep
from visipilot.target_selection.matcher import MatchCandidate, match_target
from visipilot.types import ActionOutcome, ActionRecord, Screenshot, SemanticUIState

logger = logging.getLogger("visipilot.action.runner")

# match_target's own default top_k=3 is fine for plain text matching,
# but resolving an ordinal ("the third result") needs every candidate
# genuinely tied for the top score to be visible, not just the first
# three — a duplicate-element page with more than 3 copies would
# otherwise silently truncate the pool an ordinal indexes into. Larger
# than any real page this project's test suite uses; harmless for
# non-ordinal resolution since resolve_single_candidate only ever
# inspects the top-scoring tied group regardless of pool size.
_ORDINAL_CANDIDATE_POOL = 10

StaleCheck = Callable[[Page, Screenshot], bool]
# Re-runs the real perception chain (capture -> detect -> OCR -> build)
# and returns a fresh SemanticUIState. Injected rather than imported so
# the runner keeps no dependency on the perception models — the same
# reason `stale_check` is injected.
RePerceive = Callable[[], SemanticUIState]


class RetryPolicy(BaseModel):
    """Hard, finite retry limits (Instructions.md #7: "All retries are
    bounded... No unbounded action loops under any circumstance").

    `max_attempts_per_step` counts the first attempt, so the default of 3
    is one attempt plus the two retries the Phase A pass criteria already
    specified ("maximum 2 retries per action step").
    `max_reperceptions_per_run` is a separate whole-run budget so a long
    instruction can't multiply per-step allowances into an arbitrarily
    long run.
    """

    max_attempts_per_step: int = 3
    max_reperceptions_per_run: int = 3


def run_steps(
    page: Page,
    steps: list[InstructionStep],
    state: SemanticUIState,
    stale_check: StaleCheck | None = screenshot_has_changed,
    reperceive: RePerceive | None = None,
    clarifier: Clarifier | None = None,
    policy: RetryPolicy | None = None,
) -> list[ActionRecord]:
    """Execute `steps` in order, returning one `ActionRecord` per attempt.

    Stops at the first step that can't be completed safely: no
    candidates, low confidence, an unresolved tie, a page that keeps
    changing under it, or a retry budget running out. The returned list
    is shorter than `steps` in that case, and its last record's
    `outcome`/`error_message` says why.

    `stale_check` defaults to a real, live re-screenshot comparison —
    pass `None` to disable it (e.g. in unit tests using a mocked `Page`
    that can't produce a real screenshot) or a stub for deterministic
    tests of the refusal path itself.

    `reperceive`, when given, upgrades a detected stale screenshot from
    "refuse" to "re-perceive and retry, within `policy`". When it's
    `None` (the default), staleness stays a safe refusal exactly as in
    Phase B — so the retry flow is opt-in at the call site that actually
    owns the perception models, not a hidden behaviour change.

    `clarifier`, when given, is asked which candidate was meant instead
    of refusing outright on a tie. Declining (returning `None`) is
    always honoured as a refusal.
    """
    policy = policy or RetryPolicy()
    records: list[ActionRecord] = []
    focused: MatchCandidate | None = None
    focused_phrase: str | None = None
    focused_ordinal: int | None = None
    reference_screenshot = state.screenshot
    reperceptions_used = 0

    for step in steps:
        if step.action == ActionKind.FIND:
            target, failure, clarified = _resolve("find", step.target_phrase, state, clarifier, step.ordinal)
            if failure is not None:
                records.append(failure)
                break
            focused, focused_phrase, focused_ordinal = target, step.target_phrase, step.ordinal
            records.append(
                ActionRecord(
                    action="find",
                    target_element_id=target.element.id,
                    outcome=ActionOutcome.SUCCESS,
                    clarification_used=clarified,
                )
            )
            continue

        if step.action not in (ActionKind.TYPE, ActionKind.CLICK):
            continue

        action_name = "type" if step.action == ActionKind.TYPE else "click"
        attempt = 1
        used_fresh_state = False
        step_done = False

        while not step_done:
            # --- resolve this attempt's target against the current state ---
            if step.action == ActionKind.CLICK:
                target, failure, clarified = _resolve("click", step.target_phrase, state, clarifier, step.ordinal)
            elif focused is None:
                records.append(
                    ActionRecord(
                        action="type",
                        value=step.value,
                        outcome=ActionOutcome.FAILED_NO_TARGET,
                        error_message="no prior resolved element to type into",
                        attempt=attempt,
                        reperceived=used_fresh_state,
                    )
                )
                break
            elif used_fresh_state:
                # Re-perception rebuilt every element (and every element
                # id) from scratch, so the candidate FIND resolved no
                # longer refers to anything in the current state —
                # re-resolve it from the phrase (and ordinal, if any)
                # that produced it rather than acting on a stale object.
                target, failure, clarified = _resolve("type", focused_phrase, state, clarifier, focused_ordinal)
                if target is not None:
                    focused = target
            else:
                target, failure, clarified = focused, None, False

            if failure is not None:
                failure.attempt = attempt
                failure.reperceived = used_fresh_state
                records.append(failure)
                break

            # --- staleness gate, then act ---
            if stale_check is not None and stale_check(page, reference_screenshot):
                # Exactly one record per attempt: either this attempt is
                # being retried (record the staleness that triggered it)
                # or the run is out of budget (record the terminal
                # exhaustion instead) — never both for one attempt.
                if reperceive is None:
                    # Phase B behaviour, unchanged when no re-perception
                    # capability was injected: detect and refuse.
                    records.append(_stale_record(action_name, target, step.value, attempt, used_fresh_state))
                    break
                if attempt >= policy.max_attempts_per_step:
                    records.append(_exhausted_record(
                        action_name, target, step.value, attempt, used_fresh_state,
                        f"page still changing after {attempt} attempts "
                        f"(per-step limit {policy.max_attempts_per_step})",
                    ))
                    break
                if reperceptions_used >= policy.max_reperceptions_per_run:
                    records.append(_exhausted_record(
                        action_name, target, step.value, attempt, used_fresh_state,
                        f"page changed again but this run's re-perception budget "
                        f"({policy.max_reperceptions_per_run}) is exhausted",
                    ))
                    break

                records.append(_stale_record(action_name, target, step.value, attempt, used_fresh_state))
                reperceptions_used += 1
                attempt += 1
                logger.info(
                    "stage=reperception status=start action=%s attempt=%d budget_used=%d/%d",
                    action_name, attempt, reperceptions_used, policy.max_reperceptions_per_run,
                )
                state = reperceive()
                reference_screenshot = state.screenshot
                used_fresh_state = True
                continue

            record = _execute(page, step, target, state, action_name)
            record.attempt = attempt
            record.reperceived = used_fresh_state
            record.clarification_used = clarified
            records.append(record)
            if record.outcome != ActionOutcome.SUCCESS:
                break
            if step.action == ActionKind.CLICK:
                focused, focused_phrase, focused_ordinal = target, step.target_phrase, step.ordinal
            if stale_check is not None:
                reference_screenshot = capture_screenshot(page, full_page=reference_screenshot.meta.full_page)
            step_done = True

        if not step_done:
            break

    return records


def _execute(page: Page, step: InstructionStep, target: MatchCandidate, state: SemanticUIState, action_name: str) -> ActionRecord:
    # Read meta off the *current* state, not a value captured before any
    # re-perception: a fresh perception pass carries its own screenshot
    # metadata (scroll offset above all), and grounding a click with the
    # previous pass's metadata is exactly the stale-coordinate bug this
    # milestone exists to prevent.
    meta = state.screenshot.meta
    if action_name == "click":
        return click_element(page, target, meta)
    return type_into_element(page, target, meta, step.value or "")


def _resolve(
    action: str,
    phrase: str | None,
    state: SemanticUIState,
    clarifier: Clarifier | None,
    ordinal: int | None = None,
) -> tuple[MatchCandidate | None, ActionRecord | None, bool]:
    """Resolve `phrase` to one safe candidate.

    Returns `(candidate, failure_record, clarification_used)` — exactly
    one of the first two is ever non-None.

    `ordinal`, when given, is passed straight through to
    `resolve_single_candidate` so "the second result" resolves by
    reading-order position among the tied candidates instead of refusing
    — never consulting `clarifier` in that case, since a well-formed
    ordinal that's in range doesn't need clarification, and an
    out-of-range one (`OrdinalOutOfRangeError`) is exactly as unresolvable
    by re-asking the same question as by guessing.
    """
    candidates = match_target(phrase, state, top_k=_ORDINAL_CANDIDATE_POOL)
    try:
        return resolve_single_candidate(candidates, ordinal=ordinal), None, False
    except OrdinalOutOfRangeError as exc:
        return None, _failure_record(action, exc), False
    except AmbiguousTargetError as exc:
        if clarifier is not None:
            chosen = clarifier(
                ClarificationRequest(action=action, target_phrase=phrase, candidates=exc.tied_candidates)
            )
            if chosen is not None:
                logger.info("stage=clarification status=resolved action=%s chosen=%s", action, chosen.element.id)
                return chosen, None, True
            logger.info("stage=clarification status=declined action=%s candidates=%d", action, len(exc.tied_candidates))
        return None, _failure_record(action, exc), False
    except NoConfidentTargetError as exc:
        return None, _failure_record(action, exc), False


def _failure_record(action: str, exc: Exception) -> ActionRecord:
    if isinstance(exc, AmbiguousTargetError):
        outcome = ActionOutcome.FAILED_AMBIGUOUS
    elif isinstance(exc, OrdinalOutOfRangeError):
        outcome = ActionOutcome.FAILED_ORDINAL_OUT_OF_RANGE
    else:
        outcome = ActionOutcome.FAILED_NO_TARGET
    return ActionRecord(action=action, outcome=outcome, error_message=str(exc))


def _stale_record(action: str, candidate: MatchCandidate, value: str | None, attempt: int, reperceived: bool) -> ActionRecord:
    return ActionRecord(
        action=action,
        target_element_id=candidate.element.id,
        value=value if action == "type" else None,
        outcome=ActionOutcome.FAILED_STALE_SCREENSHOT,
        error_message="page content changed since the perception screenshot was captured; refusing to act on now-unverified coordinates",
        attempt=attempt,
        reperceived=reperceived,
    )


def _exhausted_record(
    action: str, candidate: MatchCandidate, value: str | None, attempt: int, reperceived: bool, reason: str
) -> ActionRecord:
    return ActionRecord(
        action=action,
        target_element_id=candidate.element.id,
        value=value if action == "type" else None,
        outcome=ActionOutcome.FAILED_RETRY_EXHAUSTED,
        error_message=reason,
        attempt=attempt,
        reperceived=reperceived,
    )
