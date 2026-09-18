"""Clarification path for genuinely ambiguous targets (Phase C).

When target selection returns several candidates tied within the
executor's `tie_epsilon`, the system has always refused to guess
(Instructions.md #7). This module adds the other half the roadmap asks
for: a defined mechanism for *asking* which one was meant, instead of
only ever refusing.

Deliberately a plain injected callable, not a framework: the runner
takes an optional `Clarifier` exactly the way it already takes an
optional `stale_check`, so there is one injection pattern in this
codebase rather than two. The default stays `None` — with no clarifier
wired up, behaviour is bit-for-bit what it was before (safe refusal,
`FAILED_AMBIGUOUS`), so adding this path cannot silently weaken the
no-blind-clicking rule.

**Security (Instructions.md #6).** Candidate text comes from the page
and is therefore untrusted input. `cli_clarifier` prints it inside an
explicit `page text:` delimiter and never as part of a prompt the user
could mistake for a system instruction, and it only ever accepts a
numeric choice back — page content can never steer the selection, only
describe it.
"""
from __future__ import annotations

from typing import Callable

from pydantic import BaseModel

from visipilot.target_selection.matcher import MatchCandidate


class ClarificationRequest(BaseModel):
    """Everything a clarifier needs to ask a useful question, and nothing
    that would let it act on its own.
    """

    action: str  # "find" | "click" | "type"
    target_phrase: str | None
    candidates: list[MatchCandidate]


# Returns the chosen candidate, or None to decline (which the runner
# treats exactly like the pre-Phase-C behaviour: safe refusal).
Clarifier = Callable[[ClarificationRequest], MatchCandidate | None]


def format_clarification_prompt(request: ClarificationRequest) -> str:
    """Render the question as text. Split out from `cli_clarifier` so the
    wording is testable without driving stdin/stdout.
    """
    phrase = request.target_phrase or "(no phrase)"
    lines = [
        f"Ambiguous target for {request.action} {phrase!r}: "
        f"{len(request.candidates)} candidates scored within the tie threshold.",
    ]
    for i, candidate in enumerate(request.candidates, start=1):
        el = candidate.element
        bbox = el.bbox
        # Page-derived text is untrusted (Instructions.md #6): label it as
        # page content, never interpolate it as if it were instruction text.
        text = el.text if el.text else "(no text)"
        lines.append(
            f"  [{i}] id={el.id} score={candidate.score:.3f} type={el.type.value} "
            f"at ({bbox.x:.0f},{bbox.y:.0f}) {bbox.width:.0f}x{bbox.height:.0f} "
            f"page text: {text!r}"
        )
    lines.append("Choose 1-%d, or press Enter to refuse: " % len(request.candidates))
    return "\n".join(lines)


def cli_clarifier(request: ClarificationRequest) -> MatchCandidate | None:
    """Minimal interactive clarifier — the "even if simple, e.g. CLI
    prompt" mechanism Phase C calls for.

    Declines (returns None, i.e. safe refusal) on empty input, a
    non-numeric answer, an out-of-range number, or EOF/non-interactive
    stdin. Every one of those is a case where guessing would be exactly
    the blind action Instructions.md #7 forbids, so they all fail closed.
    """
    try:
        answer = input(format_clarification_prompt(request)).strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not answer.isdigit():
        return None
    choice = int(answer)
    if not 1 <= choice <= len(request.candidates):
        return None
    return request.candidates[choice - 1]
