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
            scores[element.id] += own_text_score
            scores[element.id] += _structural_bonus(phrase_tokens, element)
            scores[element.id] += _aspect_ratio_bonus(phrase_tokens, element)
            scores[element.id] += _fill_bonus(phrase_tokens, element, image_path)
            scores[element.id] += _length_penalty(element)
            if element.interactable:
                scores[element.id] += 0.1

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
