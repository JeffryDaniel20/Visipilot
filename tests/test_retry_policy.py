"""Unit tests for the bounded re-perception retry flow in
visipilot.action.runner (Phase C.1) — mocked Playwright page, synthetic
SemanticUIState, stubbed staleness/re-perception so the *policy* is
tested deterministically rather than by racing a real page.

The real-page behaviour these bounds exist for is covered separately by
tests/test_stale_screenshot.py against a genuinely mutating page.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from visipilot.action.runner import RetryPolicy, run_steps
from visipilot.target_selection.instruction_parser import parse_instruction
from visipilot.types import (
    ActionOutcome,
    BBox,
    CoordinateSpace,
    ElementSource,
    ElementType,
    Screenshot,
    ScreenshotMeta,
    SemanticUIState,
    UIElement,
)


def make_element(id_, text, x=0, y=0, w=80, h=30, etype=ElementType.BUTTON):
    return UIElement(
        id=id_,
        type=etype,
        bbox=BBox(x=x, y=y, width=w, height=h, space=CoordinateSpace.SCREENSHOT_PX),
        text=text,
        confidence=0.9,
        interactable=True,
        source=ElementSource.FUSION,
    )


def make_state(elements, image_hash="abc"):
    meta = ScreenshotMeta(viewport_width=1280, viewport_height=800)
    screenshot = Screenshot(image_path="x.png", image_hash=image_hash, width_px=1280, height_px=800, meta=meta)
    return SemanticUIState(elements=elements, screenshot=screenshot)


@pytest.fixture(autouse=True)
def stub_reference_screenshot(monkeypatch):
    """`run_steps` re-captures a cheap screenshot after each successful
    action to roll the staleness reference forward, then (C.7) checks it
    against the pre-action screenshot outside the acted-on element to
    decide whether that roll-forward is safe to trust. A MagicMock page
    can't produce real PNG bytes, and the fake `Screenshot`s these tests
    build carry no real image on disk for either call to decode, so stub
    both — the staleness decision itself is driven by the injected
    `stale_check` stub, which is what these tests are actually
    exercising, not the real pixel comparisons (those are covered
    against a genuine browser by tests/test_stale_screenshot.py).
    """
    monkeypatch.setattr(
        "visipilot.action.runner.capture_screenshot",
        lambda page, full_page=False: make_state([]).screenshot,
    )
    monkeypatch.setattr(
        "visipilot.action.runner.screenshot_changed_outside",
        lambda old, new, exclude: False,
    )


def test_stale_once_then_reperceive_succeeds():
    state = make_state([make_element("btn", "Search")])
    fresh = make_state([make_element("btn2", "Search")], image_hash="fresh")
    page = MagicMock()
    calls = {"stale": 0, "reperceive": 0}

    def stale_check(_page, _screenshot):
        calls["stale"] += 1
        return calls["stale"] == 1  # stale on the first check only

    def reperceive():
        calls["reperceive"] += 1
        return fresh

    records = run_steps(page, parse_instruction("Click Search."), state,
                        stale_check=stale_check, reperceive=reperceive)

    assert [r.outcome for r in records] == [
        ActionOutcome.FAILED_STALE_SCREENSHOT,
        ActionOutcome.SUCCESS,
    ]
    assert calls["reperceive"] == 1
    # The retry is visible per-attempt, not just in aggregate.
    assert records[0].attempt == 1 and records[0].reperceived is False
    assert records[1].attempt == 2 and records[1].reperceived is True
    # It acted on the *fresh* state's element, not the original one.
    assert records[1].target_element_id == "btn2"
    page.mouse.click.assert_called_once()


def test_permanently_changing_page_stops_at_per_step_attempt_limit():
    # The critical bound: a page that never settles must terminate, not
    # loop (Instructions.md #7 — "No unbounded action loops under any
    # circumstance").
    state = make_state([make_element("btn", "Search")])
    page = MagicMock()
    reperceive_calls = {"n": 0}

    def always_stale(_page, _screenshot):
        return True

    def reperceive():
        reperceive_calls["n"] += 1
        return make_state([make_element("btn", "Search")], image_hash=f"h{reperceive_calls['n']}")

    records = run_steps(page, parse_instruction("Click Search."), state,
                        stale_check=always_stale, reperceive=reperceive,
                        policy=RetryPolicy(max_attempts_per_step=3, max_reperceptions_per_run=10))

    assert records[-1].outcome == ActionOutcome.FAILED_RETRY_EXHAUSTED
    assert [r.attempt for r in records] == [1, 2, 3]
    assert reperceive_calls["n"] == 2  # attempts 1->2 and 2->3, then it stops
    page.mouse.click.assert_not_called()


def test_run_wide_reperception_budget_caps_a_multi_step_instruction():
    # Per-step limits alone would let N steps multiply into N x M
    # perception passes; the whole-run budget is what actually bounds
    # total work.
    state = make_state([
        make_element("inp", "Search:", w=300, etype=ElementType.TEXT_INPUT),
        make_element("btn", "Search", x=400),
    ])
    page = MagicMock()
    reperceive_calls = {"n": 0}

    def reperceive():
        reperceive_calls["n"] += 1
        return state

    records = run_steps(page, parse_instruction("Find the search box, type Python, and click Search."), state,
                        stale_check=lambda _p, _s: True, reperceive=reperceive,
                        policy=RetryPolicy(max_attempts_per_step=5, max_reperceptions_per_run=2))

    assert records[-1].outcome == ActionOutcome.FAILED_RETRY_EXHAUSTED
    assert "budget" in records[-1].error_message
    assert reperceive_calls["n"] == 2  # never exceeds the run-wide budget


def test_without_reperceive_staleness_stays_a_plain_refusal():
    # Phase B behaviour must be bit-for-bit preserved when no
    # re-perception capability is injected -- the retry flow is opt-in,
    # never a silent behaviour change for existing callers.
    state = make_state([make_element("btn", "Search")])
    page = MagicMock()

    records = run_steps(page, parse_instruction("Click Search."), state,
                        stale_check=lambda _p, _s: True, reperceive=None)

    assert [r.outcome for r in records] == [ActionOutcome.FAILED_STALE_SCREENSHOT]
    assert records[0].attempt == 1
    page.mouse.click.assert_not_called()


def test_no_retry_happens_when_page_is_stable():
    # A static page must cost exactly zero extra perception passes --
    # the retry path is free when it isn't needed.
    state = make_state([make_element("btn", "Search")])
    page = MagicMock()
    reperceive_calls = {"n": 0}

    def reperceive():
        reperceive_calls["n"] += 1
        return state

    records = run_steps(page, parse_instruction("Click Search."), state,
                        stale_check=lambda _p, _s: False, reperceive=reperceive)

    assert [r.outcome for r in records] == [ActionOutcome.SUCCESS]
    assert records[0].attempt == 1 and records[0].reperceived is False
    assert reperceive_calls["n"] == 0


def test_type_step_reresolves_its_target_against_freshly_perceived_state():
    # Re-perception rebuilds every element id from scratch, so a TYPE
    # step must re-resolve the element FIND had focused rather than act
    # on an object that no longer refers to anything on the page.
    original = make_state([make_element("inp-old", "Search:", w=300, etype=ElementType.TEXT_INPUT)])
    fresh = make_state([make_element("inp-new", "Search:", w=300, etype=ElementType.TEXT_INPUT)], image_hash="fresh")
    page = MagicMock()
    checks = {"n": 0}

    def stale_check(_page, _screenshot):
        checks["n"] += 1
        return checks["n"] == 1

    records = run_steps(page, parse_instruction("Find the search box, type Python."), original,
                        stale_check=stale_check, reperceive=lambda: fresh)

    assert [r.outcome for r in records] == [
        ActionOutcome.SUCCESS,                      # find, against the original state
        ActionOutcome.FAILED_STALE_SCREENSHOT,      # type, attempt 1
        ActionOutcome.SUCCESS,                      # type, attempt 2 after re-perception
    ]
    assert records[0].target_element_id == "inp-old"
    assert records[2].target_element_id == "inp-new"
    page.keyboard.type.assert_called_once_with("Python")


def test_post_action_external_change_forces_resync_before_next_step(monkeypatch):
    """C.7: the root-cause scenario, reproduced deterministically.

    Before this fix, `reference_screenshot` was rolled forward to a
    fresh, unchecked capture after every successful action, purely to
    absorb that action's own expected visual delta (e.g. TYPE's echoed
    text) so the *next* step's staleness check wouldn't see a false
    positive. But a fresh capture reflects *everything* live about the
    page at that instant, not just the acted-on element's own change --
    so if an unrelated external change (e.g. a late banner) happened to
    land in the same instant, without either step's own pre-action
    check ever independently catching it as stale (simulated here via
    `stale_check` that always reports "not stale"), it got silently
    absorbed into the new baseline while `state` -- the source of the
    *next* step's coordinates -- was never told anything happened.

    This is exactly what `screenshot_changed_outside` now exists to
    catch: even though neither step's own pre-action check fires here,
    the post-action check (stubbed to report "something changed outside
    the typed element") must force a resync before CLICK resolves its
    target, so CLICK acts on the fresh element (`btn-new`) rather than
    the stale one (`btn-old`).
    """
    original = make_state([
        make_element("inp", "Search:", w=300, etype=ElementType.TEXT_INPUT),
        make_element("btn-old", "Submit", x=400),
    ])
    fresh = make_state([
        make_element("inp", "Search:", w=300, etype=ElementType.TEXT_INPUT),
        make_element("btn-new", "Submit", x=400, y=60),
    ], image_hash="fresh")
    page = MagicMock()

    monkeypatch.setattr(
        "visipilot.action.runner.screenshot_changed_outside",
        lambda old, new, exclude: True,
    )
    reperceive_calls = {"n": 0}

    def reperceive():
        reperceive_calls["n"] += 1
        return fresh

    records = run_steps(
        page, parse_instruction("Find the search box, type Python, and click Submit."), original,
        stale_check=lambda _p, _s: False,
        reperceive=reperceive,
    )

    assert [r.outcome for r in records] == [
        ActionOutcome.SUCCESS,  # find
        ActionOutcome.SUCCESS,  # type
        ActionOutcome.SUCCESS,  # click
    ]
    # Fires once after TYPE (the resync this test exists to prove) and
    # once more after CLICK -- this stub reports "changed outside" on
    # every call, and the fix does not look ahead to see CLICK is the
    # run's last step, so it resyncs defensively there too. Harmless
    # (nothing downstream depends on that second resync) and correctly
    # bounded by the same run-wide budget as any other re-perception.
    assert reperceive_calls["n"] == 2
    assert records[2].target_element_id == "btn-new"


def test_post_action_external_change_refuses_without_reperceive(monkeypatch):
    """Same scenario as above, but with no `reperceive` injected: safe
    refusal, not a silent click on stale coordinates -- matching the
    Phase B behaviour a pre-action staleness detection already gets
    when no re-perception capability is wired up.
    """
    original = make_state([
        make_element("inp", "Search:", w=300, etype=ElementType.TEXT_INPUT),
        make_element("btn-old", "Submit", x=400),
    ])
    page = MagicMock()

    monkeypatch.setattr(
        "visipilot.action.runner.screenshot_changed_outside",
        lambda old, new, exclude: True,
    )

    records = run_steps(
        page, parse_instruction("Find the search box, type Python, and click Submit."), original,
        stale_check=lambda _p, _s: False,
        reperceive=None,
    )

    assert [r.outcome for r in records] == [
        ActionOutcome.SUCCESS,               # find
        ActionOutcome.SUCCESS,               # type
        ActionOutcome.FAILED_RETRY_EXHAUSTED,  # resync needed but unavailable -- refuse
    ]
    # TYPE itself legitimately clicks the input once, to focus it, before
    # typing -- what must never happen is a *second* click, on the
    # (possibly wrong) "Submit" button, which would be exactly the
    # blind click on stale coordinates this whole fix exists to prevent.
    assert page.mouse.click.call_count == 1
