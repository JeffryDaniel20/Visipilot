"""Visual signal extraction for target-selection scoring: samples the
real screenshot's pixel color inside a candidate element's bbox.

Pixels-first by construction — reads the same screenshot image the
perception stage already produced (via `SemanticUIState.screenshot`),
never the DOM and never a detector-provided color/style attribute (there
isn't one). Kept as a separate module from matcher.py because it does
image I/O, a different concern from pure text scoring.
"""
from __future__ import annotations

from functools import lru_cache

from PIL import Image, ImageStat

from visipilot.types import BBox

# Full-bonus distance: empirically, this project's own real button fills
# (blue #2b5cff, gray #e0e0e0) measured 293px and 116px from white
# respectively, against real input fills (white #fff, near-white #fbfbfb)
# measuring ~0-4px. 120 sits below the weaker (gray) real button's
# measurement and well above real input measurements — see
# implementation-plan.md for the measurement this threshold is based on.
FULL_BONUS_DISTANCE = 120.0


@lru_cache(maxsize=8)
def _load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def fill_distance_from_white(image_path: str, bbox: BBox) -> float:
    """Average RGB Euclidean distance from pure white (0 to ~441),
    sampled from the bbox's interior (inset 25% on each side to avoid
    border pixels and text glyphs skewing the average toward black).

    Real text-entry controls are overwhelmingly styled with a white or
    near-white fill (the default browser input style, and the common
    convention of keeping inputs light for contrast with typed text);
    real buttons — whether brand-colored or a neutral gray secondary
    style — are reliably some visible distance from pure white.

    This was verified empirically against four real, differently-styled
    elements on this project's own test page before being adopted, not
    assumed: a naive "high color saturation = button" version was tried
    first and rejected because it fails on the page's gray Subscribe
    button (saturation 0.004, indistinguishable from white's 0.000) —
    distance-from-white separates all four correctly (search input ~0,
    newsletter input ~4, gray Subscribe button ~116, blue Search button
    ~293). See implementation-plan.md for the raw measurement.

    Known, stated limitation: this will not generalize to "ghost"/outline
    buttons with a white or transparent fill, or to dark-mode pages where
    inputs may not be styled white. Not fixed here — flagged as a Phase B
    item, to be measured against real dark-mode/varied-styling test pages
    once they exist, rather than guessed at now.
    """
    image = _load_image(image_path)
    x0 = max(0, int(bbox.x + bbox.width * 0.25))
    y0 = max(0, int(bbox.y + bbox.height * 0.25))
    x1 = min(image.width, int(bbox.x + bbox.width * 0.75))
    y1 = min(image.height, int(bbox.y + bbox.height * 0.75))
    if x1 <= x0 or y1 <= y0:
        return 0.0

    crop = image.crop((x0, y0, x1, y1))
    avg_r, avg_g, avg_b = ImageStat.Stat(crop).mean
    return ((255 - avg_r) ** 2 + (255 - avg_g) ** 2 + (255 - avg_b) ** 2) ** 0.5
