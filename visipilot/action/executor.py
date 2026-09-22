"""Action Executor: issues real Playwright click/type actions using
grounded target coordinates and trusted input events (`page.mouse`,
`page.keyboard` — never JS-injected clicks).

Never picks a target on its own: `resolve_single_candidate` enforces
Instructions.md #7 — a low-confidence or tied top candidate must not be
blindly acted on. The caller (or, eventually, a clarification flow in
Phase C) decides what to do when that happens; this module only refuses.
"""
from __future__ import annotations

from playwright.sync_api import Page

from visipilot.grounding.coordinates import bbox_to_viewport_click_point, is_within_viewport
from visipilot.target_selection.matcher import MatchCandidate
from visipilot.types import ActionOutcome, ActionRecord, ScreenshotMeta

DEFAULT_MIN_CONFIDENCE = 0.3
DEFAULT_TIE_EPSILON = 0.05


class NoConfidentTargetError(Exception):
    """Raised when there are no candidates, or the top one scores below
    the minimum confidence threshold.
    """


class AmbiguousTargetError(Exception):
    """Raised when the top two (or more) candidates are tied within
    `tie_epsilon` — genuine ambiguity, not a single safe answer.
    """

    def __init__(self, tied_candidates: list[MatchCandidate]):
        self.tied_candidates = tied_candidates
        super().__init__(
            f"{len(tied_candidates)} candidates tied within epsilon, refusing to guess: "
            + ", ".join(f"{c.element.id}({c.score:.3f})" for c in tied_candidates)
        )


class OrdinalOutOfRangeError(Exception):
    """Raised when an instruction referenced an ordinal position (e.g.
    "the second result") that doesn't exist among the tied candidates —
    e.g. asking for the third of only two matches. Distinct from
    `AmbiguousTargetError`: the tie itself isn't the problem here (an
    ordinal is exactly the disambiguating information a genuine tie
    needs), the requested position just doesn't exist. Refusing is still
    the only safe response — picking the closest existing position would
    be exactly the kind of guess Instructions.md #7 forbids.
    """

    def __init__(self, ordinal: int, available: int):
        self.ordinal = ordinal
        self.available = available
        super().__init__(
            f"ordinal {ordinal} requested but only {available} candidate(s) tied for this phrase"
        )


def _resolve_ordinal(tied: list[MatchCandidate], ordinal: int) -> MatchCandidate:
    """Pick the `ordinal`-th (1-based; -1 means "last") of `tied` by
    reading order — top-to-bottom, then left-to-right by bbox center.

    Reading order is the only ordering pixels-first perception can
    determine at all (Instructions.md #2: no DOM access at runtime), and
    it's also what a human means by "the first/second result" for the
    vertically- or horizontally-stacked layouts real duplicate-element
    pages actually use — not DOM/HTML source order, which a pixels-first
    matcher never sees and which can legitimately differ from what's
    visually first (e.g. CSS `order`, reversed flex direction).
    """
    ordered = sorted(tied, key=lambda c: (c.element.bbox.center[1], c.element.bbox.center[0]))
    n = len(ordered)
    index = ordinal - 1 if ordinal > 0 else ordinal
    if not -n <= index < n:
        raise OrdinalOutOfRangeError(ordinal, n)
    return ordered[index]


def resolve_single_candidate(
    candidates: list[MatchCandidate],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    tie_epsilon: float = DEFAULT_TIE_EPSILON,
    ordinal: int | None = None,
) -> MatchCandidate:
    """Return the single candidate that's safe to act on, or raise.

    `candidates` must already be sorted descending by score (as
    `match_target` returns them).

    `ordinal`, when given, resolves a tie by reading-order position
    (e.g. "the second search button") instead of refusing — but only
    within the group of candidates actually tied for the top score;
    an ordinal never reaches past a genuine top-score group into
    lower-scored, less-relevant matches, and an out-of-range ordinal
    (e.g. "third" when only two are tied) still refuses via
    `OrdinalOutOfRangeError` rather than guessing.
    """
    if not candidates:
        raise NoConfidentTargetError("no candidates returned by target selection")
    top = candidates[0]
    if top.score < min_confidence:
        raise NoConfidentTargetError(
            f"top candidate {top.element.id!r} scored {top.score:.3f}, below min_confidence {min_confidence}"
        )
    tied = [c for c in candidates if (top.score - c.score) <= tie_epsilon]
    if ordinal is not None:
        return _resolve_ordinal(tied, ordinal)
    if len(tied) > 1:
        raise AmbiguousTargetError(tied)
    return top


def _scroll_into_view_and_recompute(
    page: Page, candidate: MatchCandidate, meta: ScreenshotMeta, click_x: float, click_y: float
) -> tuple[float, float] | None:
    """One bounded, trusted-input attempt to bring an out-of-viewport
    click point into view, then recompute it — or `None` if it's still
    out of view afterward (never retried again, never guessed at).

    Only meaningful when perception captured the whole scrollable page
    (`meta.full_page`): that's the only case where a target below/above
    the current viewport could have been perceived at all, and the only
    case `bbox_to_viewport_click_point` even needs a scroll offset for.
    Scrolls via `page.mouse.wheel()` — a real trusted wheel event, same
    class of input as the `page.mouse.click()`/`page.keyboard.type()`
    this module already exclusively uses, never a JS-injected
    `window.scrollTo()`. The delta is only an estimate (real browsers
    don't guarantee wheel-delta-to-scroll-pixel is 1:1, and may animate
    the scroll) — `meta.scroll_y` is updated from a live, pixels-first-
    safe *read* of `window.scrollY` (the same mechanism `capture_screenshot`
    already uses to record scroll offset, not a DOM query for the
    target's position) after scrolling settles, and the click point is
    recomputed from that real value, not the estimate. Mutates `meta` in
    place so every later step sharing this `SemanticUIState` also sees
    the page's new scroll position, exactly as `state`/`reference_screenshot`
    must always agree with reality (Instructions.md #7 / C.7).
    """
    if not meta.full_page:
        return None
    delta_y = click_y - meta.viewport_height / 2
    if delta_y == 0:
        return None
    page.mouse.wheel(0, delta_y)
    page.wait_for_timeout(100)
    meta.scroll_y = page.evaluate("window.scrollY")
    new_x, new_y = bbox_to_viewport_click_point(candidate.element.bbox, meta)
    if not is_within_viewport(new_x, new_y, meta):
        return None
    return new_x, new_y


def click_element(page: Page, candidate: MatchCandidate, meta: ScreenshotMeta) -> ActionRecord:
    click_x, click_y = bbox_to_viewport_click_point(candidate.element.bbox, meta)
    if not is_within_viewport(click_x, click_y, meta):
        recovered = _scroll_into_view_and_recompute(page, candidate, meta, click_x, click_y)
        if recovered is None:
            return ActionRecord(
                action="click",
                target_element_id=candidate.element.id,
                click_point=(click_x, click_y),
                outcome=ActionOutcome.FAILED_OUT_OF_VIEWPORT,
            )
        click_x, click_y = recovered
    try:
        page.mouse.click(click_x, click_y)
    except Exception as exc:  # noqa: BLE001 — any Playwright failure is a legitimate action failure
        return ActionRecord(
            action="click",
            target_element_id=candidate.element.id,
            click_point=(click_x, click_y),
            outcome=ActionOutcome.FAILED_EXECUTION_ERROR,
            error_message=str(exc),
        )
    return ActionRecord(
        action="click",
        target_element_id=candidate.element.id,
        click_point=(click_x, click_y),
        outcome=ActionOutcome.SUCCESS,
    )


def type_into_element(page: Page, candidate: MatchCandidate, meta: ScreenshotMeta, text: str) -> ActionRecord:
    """Clicks the element first to focus it (trusted click, same as
    `click_element`), clears any existing content, then types via
    `page.keyboard.type` — a trusted keyboard event stream throughout,
    never a JS value assignment.

    Clearing first matters for real correctness, not just tidiness
    (implementation-plan.md C.11): confirmed directly against a real
    browser that clicking an already-filled input and typing without
    clearing it first *appends* to the existing value rather than
    replacing it (a real input holding "OldValue", clicked and typed
    "Python" into with no clear step, ends up "OldValuePython" — not
    "Python"). A pre-filled field is a common real condition (a
    default/placeholder-adjacent value, browser autofill, or simply
    re-running an instruction against a field a prior step already
    touched), and Playwright's own `page.mouse.click()` +
    `page.keyboard.type()` never clears on its own — every serious
    browser-automation tool clears/selects-all before typing for
    exactly this reason. `Control+A` (selects the focused field's own
    text, standard browser behaviour, not the whole page) then
    `Delete` are both trusted keyboard events, the same class of input
    this function already exclusively uses — never `el.value = ''`.
    """
    click_x, click_y = bbox_to_viewport_click_point(candidate.element.bbox, meta)
    if not is_within_viewport(click_x, click_y, meta):
        recovered = _scroll_into_view_and_recompute(page, candidate, meta, click_x, click_y)
        if recovered is None:
            return ActionRecord(
                action="type",
                target_element_id=candidate.element.id,
                click_point=(click_x, click_y),
                value=text,
                outcome=ActionOutcome.FAILED_OUT_OF_VIEWPORT,
            )
        click_x, click_y = recovered
    try:
        page.mouse.click(click_x, click_y)
        page.keyboard.press("Control+A")
        page.keyboard.press("Delete")
        page.keyboard.type(text)
    except Exception as exc:  # noqa: BLE001
        return ActionRecord(
            action="type",
            target_element_id=candidate.element.id,
            click_point=(click_x, click_y),
            value=text,
            outcome=ActionOutcome.FAILED_EXECUTION_ERROR,
            error_message=str(exc),
        )
    return ActionRecord(
        action="type",
        target_element_id=candidate.element.id,
        click_point=(click_x, click_y),
        value=text,
        outcome=ActionOutcome.SUCCESS,
    )
