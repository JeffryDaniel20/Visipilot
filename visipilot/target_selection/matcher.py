"""Target Selection matcher: rank UIElements in a SemanticUIState against
a target phrase, using text + semantic_role + relation + visual-fill
signals. Reads the same screenshot image the perception stage already
produced (via `state.screenshot.image_path`) when a text-only signal
can't disambiguate — allowed by Instructions.md #2's actual pixels-first
rule ("perception ... and target selection must operate only on
screenshot pixels"), never the DOM/accessibility tree.
"""
from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel

from visipilot.target_selection.visual_signals import FULL_BONUS_DISTANCE, fill_distance_from_white
from visipilot.types import SemanticUIState, UIElement

_STOPWORDS = {"the", "a", "an", "on"}
_STRUCTURAL_WORDS = {"box", "field", "input"}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class MatchCandidate(BaseModel):
    element: UIElement
    score: float


def _text_score(phrase_tokens: set[str], element: UIElement) -> float:
    text_tokens = _tokenize(element.text or "")
    if not text_tokens:
        return 0.0
    meaningful_phrase = phrase_tokens - _STOPWORDS or phrase_tokens
    overlap = meaningful_phrase & text_tokens
    if not overlap:
        return 0.0
    score = len(overlap) / len(meaningful_phrase)
    if meaningful_phrase == text_tokens:
        score += 0.5  # exact match bonus
    return score


def _length_penalty(element: UIElement) -> float:
    """Real UI control labels (button text, placeholders, short captions)
    are almost never more than a handful of words. A long, paragraph-like
    text field scoring well on token overlap is far more likely to be
    incidental content that happens to share words with the target
    phrase than an actual match — discovered honestly: the controlled
    test page's own distractor paragraph describes "the search box" in
    its body text and, without this penalty, out-scored the real search
    input on that exact phrase (see implementation-plan.md).
    """
    word_count = len((element.text or "").split())
    return -1.0 if word_count > 6 else 0.0


def _structural_bonus(phrase_tokens: set[str], element: UIElement) -> float:
    """A phrase using "box"/"field"/"input" implies a data-entry control,
    not a button — nudge away from short, imperative-looking text like
    "Search"/"Submit"/"Go" (single-word element text).
    """
    if not (phrase_tokens & _STRUCTURAL_WORDS):
        return 0.0
    text = (element.text or "").strip()
    if text and len(text.split()) <= 1:
        return 0.0
    return 0.15


_INPUT_LIKE_ASPECT_RATIO = 3.0
_ASPECT_RATIO_BONUS = 0.2


def _aspect_ratio_bonus(phrase_tokens: set[str], element: UIElement) -> float:
    """A second, geometry-based signal for the same "box"/"field"/"input"
    phrases `_structural_bonus` targets — added after real-pipeline
    testing showed `_structural_bonus` alone doesn't help when both
    candidates' OCR text is a single token (e.g. "Search:" vs "Search",
    both one "word" by whitespace splitting once EasyOCR merges the
    placeholder ellipsis into a colon). Real text-entry controls are
    reliably much wider than they are tall; buttons are not. This is a
    general UI-layout heuristic, not tuned to this project's specific
    test page pixel values.
    """
    if not (phrase_tokens & _STRUCTURAL_WORDS):
        return 0.0
    bbox = element.bbox
    if bbox.height <= 0:
        return 0.0
    aspect_ratio = bbox.width / bbox.height
    return _ASPECT_RATIO_BONUS if aspect_ratio >= _INPUT_LIKE_ASPECT_RATIO else 0.0


_FILL_BONUS_MAX = 0.3


def _fill_bonus(phrase_tokens: set[str], element: UIElement, image_path: str | None) -> float:
    """Real UI buttons — colored or neutral — reliably have a fill some
    distance from pure white; real text inputs are reliably close to
    white. See `visual_signals.fill_distance_from_white` for the
    measurement this is based on.

    Deliberately gated OFF when the phrase already contains a structural
    word ("box"/"field"/"input"): `_aspect_ratio_bonus` already resolves
    that case correctly (real evidence: it disambiguates "the search
    box" in favor of the real input), and adding this bonus there too
    would push the button's score back up and re-introduce a tie this
    project already fixed once — verified by hand-computing both cases
    before wiring this in, not assumed.
    """
    if phrase_tokens & _STRUCTURAL_WORDS:
        return 0.0
    if not image_path:
        return 0.0
    try:
        distance = fill_distance_from_white(image_path, element.bbox)
    except (OSError, ValueError):
        # A missing/corrupt screenshot file must never break matching —
        # this signal is a refinement, not a required input.
        return 0.0
    return min(1.0, distance / FULL_BONUS_DISTANCE) * _FILL_BONUS_MAX


_NEIGHBOR_MAX_OVERLAP_RATIO = 0.1
# Real icon-only controls are compact (a magnifying-glass/clear/toggle
# icon is typically 16-48px). This cap was added after a real false
# positive was found testing this feature: on search_bootstrap.html, a
# large (375x96) spurious detector box with no fused text was a
# non-overlapping `nearby` neighbor of the real Search button purely by
# edge-gap distance, and without a size cap it was wrongly treated as an
# "adjacent icon" and redirected to — creating a NEW wrong click that
# didn't exist before this feature. A large region is not what "icon
# button beside an input" means, however close it happens to sit.
_NEIGHBOR_MAX_DIMENSION_PX = 48.0


def _bbox_overlap_ratio(a, b) -> float:
    """Fraction of the SMALLER box's area that overlaps with the other —
    deliberately relative to `min(area_a, area_b)`, not just one box's
    area: dividing by a single fixed box's area misses the case where a
    small box is fully swallowed by a much larger one (a real bug found
    testing this: a large spurious detector box gave a *low* overlap
    ratio purely because its own area was huge, even though it entirely
    contained the small element being checked). Local to this module —
    fusion.py and relations.py each have their own small, purpose-fit
    copy of similar geometry rather than a shared import.
    """
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    smaller_area = min(a.width * a.height, b.width * b.height)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def _edge_gap(a, b) -> float:
    dx = max(a.x - b.x2, b.x - a.x2, 0.0)
    dy = max(a.y - b.y2, b.y - a.y2, 0.0)
    return (dx**2 + dy**2) ** 0.5


def _closest_textless_interactable_neighbor(element: UIElement, elements_by_id: dict[str, UIElement]) -> UIElement | None:
    """Among `element`'s `nearby` relations, find the closest interactable
    neighbor that has no OCR-derived text of its own — the real pattern
    an icon-only button (e.g. a magnifying-glass search icon with no
    visible label) next to its input forms, and which is otherwise
    completely invisible to text-based matching: it can never be found
    by its own text (it has none), so a text match on the input beside
    it is currently the *only* thing that can accidentally match the
    right region of the page.

    Three things were wrong in earlier versions of this, each found by
    testing against real pages before trusting the fix — not assumed:
    (1) "closest" by bbox-center distance picked a large, spurious
    detector box that heavily *overlapped* the input, not the genuinely
    adjacent icon button — a big overlapping box's center can be closer
    than a small adjacent box's center purely due to size, which isn't
    what "adjacent" means. Fixed by excluding neighbors that overlap
    `element` by more than a small tolerance (heavy overlap is a sign of
    a second, spurious detection of roughly the same region, not a
    distinct nearby control — the same failure pattern fixed in
    `visipilot/perception/fusion.py`'s best-overlap-match fix). (2) Once
    overlap is excluded, ranking must use edge-to-edge gap (what `nearby`
    itself is computed from), not center distance, so a large-but-truly-
    adjacent box isn't penalized just for being physically larger. (3)
    Even with (1) and (2), re-testing against `search_bootstrap.html`
    (a page with no icon-button ambiguity at all) found a NEW false
    positive: a large (375x96), non-overlapping, textless detector box
    happened to sit within the `nearby` distance threshold of the real
    Search button and was wrongly treated as an "adjacent icon,"
    redirecting to it and producing a wrong click that didn't exist
    before this feature. Real icon-only controls are compact — fixed by
    also requiring both dimensions to be at or under
    `_NEIGHBOR_MAX_DIMENSION_PX`, which a large stray region will not be.

    Returns None when there's no such neighbor, which is the common case.
    """
    candidates = [
        elements_by_id[rel.target_id]
        for rel in element.relations
        if rel.kind == "nearby"
        and rel.target_id in elements_by_id
        and elements_by_id[rel.target_id].interactable
        and not elements_by_id[rel.target_id].text
        and elements_by_id[rel.target_id].bbox.width <= _NEIGHBOR_MAX_DIMENSION_PX
        and elements_by_id[rel.target_id].bbox.height <= _NEIGHBOR_MAX_DIMENSION_PX
        and _bbox_overlap_ratio(element.bbox, elements_by_id[rel.target_id].bbox) <= _NEIGHBOR_MAX_OVERLAP_RATIO
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda n: _edge_gap(element.bbox, n.bbox))


def match_target(phrase: str, state: SemanticUIState, top_k: int = 3) -> list[MatchCandidate]:
    """Rank elements in `state` against `phrase`. Returns up to `top_k`
    candidates sorted by descending score.

    Callers decide what confidence separates "safe to act on" from
    "ambiguous — surface candidates instead of guessing" (Instructions.md
    #7); this function only ranks, it never picks for you.
    """
    phrase_tokens = _tokenize(phrase)
    elements_by_id = {el.id: el for el in state.elements}
    scores: dict[str, float] = defaultdict(float)
    image_path = state.screenshot.image_path if state.screenshot else None

    for element in state.elements:
        own_text_score = _text_score(phrase_tokens, element)
        if own_text_score > 0:
            # Bonuses only ever refine an already-relevant match; being
            # interactable or phrase-structural must never manufacture a
            # candidate out of zero text relevance.
            total = own_text_score
            total += _structural_bonus(phrase_tokens, element)
            total += _aspect_ratio_bonus(phrase_tokens, element)
            total += _fill_bonus(phrase_tokens, element, image_path)
            total += _length_penalty(element)
            if element.interactable:
                total += 0.1
            scores[element.id] += total

            # A textless interactable control right next to this matched
            # element (e.g. an icon-only button beside its search input)
            # is otherwise invisible to text-based matching entirely — it
            # has no text of its own to ever be found by. Rather than
            # guess which of the two a bare phrase like "Search" means (a
            # real, currently-unresolvable ambiguity — OWLv2's type field
            # is unreliable, per implementation-plan.md A.2), tie its
            # score to the matched element's so resolve_single_candidate()
            # correctly refuses instead of confidently clicking the wrong
            # one. Gated off for structural-word phrases ("the search
            # box"): those already resolve correctly via
            # _aspect_ratio_bonus/_fill_bonus, and tying here would
            # re-introduce a tie that fix already removed — verified by
            # re-running match_target("the search box") by hand before
            # adopting this gate, not assumed.
            if not (phrase_tokens & _STRUCTURAL_WORDS):
                neighbor = _closest_textless_interactable_neighbor(element, elements_by_id)
                if neighbor is not None:
                    scores[neighbor.id] += total

        # A label's text match also lends support to the interactable
        # control it labels — relevant when fusion left a standalone
        # (non-overlapping) OCR caption next to its control instead of
        # absorbing the text directly into the control's own bbox.
        for rel in element.relations:
            if rel.kind == "label_of" and rel.target_id in elements_by_id and own_text_score > 0:
                scores[rel.target_id] += own_text_score * 0.8

    candidates = [
        MatchCandidate(element=elements_by_id[eid], score=score)
        for eid, score in scores.items()
        if score > 0
    ]
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]
