"""Target Selection matcher: rank UIElements in a SemanticUIState against
a target phrase, using text + semantic_role + relation signals only —
never raw pixels or the DOM (Instructions.md #2 pixels-first rule; this
stage consumes the already-built SemanticUIState, nothing else).
"""
from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel

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

    for element in state.elements:
        own_text_score = _text_score(phrase_tokens, element)
        if own_text_score > 0:
            # Bonuses only ever refine an already-relevant match; being
            # interactable or phrase-structural must never manufacture a
            # candidate out of zero text relevance.
            scores[element.id] += own_text_score
            scores[element.id] += _structural_bonus(phrase_tokens, element)
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
