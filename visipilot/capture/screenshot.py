"""Playwright-driven screenshot capture.

Owns viewport size, device_scale_factor, and (eventually) browser zoom
configuration — see the "Screenshot Capture" module boundary in
README.md. Produces a typed Screenshot record; contains no perception
logic and no DOM-querying beyond reading back the viewport/DPR/scroll
values needed to build correct coordinate-space metadata.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageDraw
from playwright.sync_api import Page, sync_playwright

from visipilot.types import BBox, Screenshot, ScreenshotMeta

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent.parent / "out" / "screenshots"

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Thumbnail grid `screenshot_changed_outside` downsamples to before
# comparing, and the per-cell RGB-distance threshold above which a
# thumbnail cell counts as "really changed" rather than rendering noise.
# Both calibrated against real, repeated measurements (implementation-
# plan.md C.7): self-caused-only diffs (a Chromium text-rendering-mode
# shift triggered by typing, elsewhere on the page) peaked at 63/765
# across 5 real-browser runs; a real appearing element (a banner) peaked
# at 337/765 across 5 real-browser runs -- both stable, not noisy, so
# 150 sits with real margin on both sides of that measured gap.
_DIFF_THUMBNAIL_SIZE = (48, 30)
_EXTERNAL_CHANGE_THRESHOLD = 150


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _png_dimensions(data: bytes) -> tuple[int, int]:
    """Read width/height straight out of the PNG IHDR chunk.

    Avoids adding a Pillow dependency just to read two integers.
    """
    if data[:8] != _PNG_SIGNATURE:
        raise ValueError("screenshot bytes are not a PNG image")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height


@contextmanager
def launch_page(
    url: str,
    viewport_width: int = 1280,
    viewport_height: int = 800,
    device_scale_factor: float = 1.0,
    channel: str = "chrome",
    zoom: float = 1.0,
) -> Iterator[Page]:
    """Launch a browser and open ``url`` with a fixed viewport/DPR.

    Uses the system-installed Chrome via Playwright's ``channel`` option
    rather than Playwright's own bundled Chromium download — avoids an
    extra large download (Phase 0 already hit real download instability
    in this environment) and matches the Chrome 152 install already
    verified present on this machine. Falls back to Playwright's bundled
    Chromium (``channel=None``) if the caller passes one explicitly.

    ``zoom``, when not 1.0, applies a real browser page-zoom via the
    Chrome DevTools Protocol's ``Emulation.setPageScaleFactor`` — the
    same class of zoom a user's Ctrl+/Ctrl- triggers (magnifies rendered
    content without changing the CSS layout box model — confirmed
    empirically: `getBoundingClientRect()` stays unchanged while the
    screenshot's rendered pixels visibly scale), distinct from
    ``device_scale_factor`` (DPR, a hardware/display property).
    Playwright has no first-class zoom API, so this drops to a raw CDP
    session — the smallest mechanism that does this for real, not a
    simulated/synthetic stand-in. Read back via `capture_screenshot`'s
    `zoom_factor` (from `window.visualViewport.scale`, itself verified
    against real DOM ground truth at zoom 1.25/1.5 — see
    implementation-plan.md B.10).
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel) if channel else p.chromium.launch()
        context = browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            device_scale_factor=device_scale_factor,
        )
        page = context.new_page()
        page.goto(url)
        if zoom != 1.0:
            cdp = context.new_cdp_session(page)
            cdp.send("Emulation.setPageScaleFactor", {"pageScaleFactor": zoom})
        try:
            yield page
        finally:
            context.close()
            browser.close()


def capture_screenshot(
    page: Page,
    out_dir: Path = DEFAULT_OUT_DIR,
    full_page: bool = False,
) -> Screenshot:
    """Capture a screenshot of ``page`` and wrap it in a typed Screenshot
    record with explicit coordinate-space metadata.

    Viewport size, device_scale_factor, and zoom_factor are all read
    back from the live page/browser rather than accepted as parameters
    here, so they can never drift out of sync with what was actually
    configured (single source of truth).

    zoom_factor comes from `window.visualViewport.scale` — confirmed
    empirically to read 1.0 by default regardless of DPR, and to
    correctly reflect a real CDP `Emulation.setPageScaleFactor` zoom
    applied via `launch_page(..., zoom=...)` — see implementation-plan.md
    B.10 for the live-browser verification against real DOM ground
    truth that replaced the previous hardcoded-1.0 placeholder.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    viewport = page.viewport_size
    if viewport is None:
        raise RuntimeError("page has no viewport set; configure it on the browser context")

    image_bytes = page.screenshot(full_page=full_page)
    image_hash = _hash_bytes(image_bytes)
    image_path = out_dir / f"{image_hash}.png"
    image_path.write_bytes(image_bytes)

    width_px, height_px = _png_dimensions(image_bytes)

    device_scale_factor = page.evaluate("window.devicePixelRatio")
    zoom_factor = page.evaluate("window.visualViewport ? window.visualViewport.scale : 1.0")
    scroll_x = page.evaluate("window.scrollX")
    scroll_y = page.evaluate("window.scrollY")

    meta = ScreenshotMeta(
        viewport_width=viewport["width"],
        viewport_height=viewport["height"],
        device_scale_factor=device_scale_factor,
        zoom_factor=zoom_factor,
        scroll_x=scroll_x,
        scroll_y=scroll_y,
        full_page=full_page,
    )

    return Screenshot(
        image_path=str(image_path),
        image_hash=image_hash,
        width_px=width_px,
        height_px=height_px,
        meta=meta,
    )


def screenshot_has_changed(page: Page, expected: Screenshot) -> bool:
    """Cheap re-screenshot + hash compare against `expected` — the
    mechanism behind Instructions.md #7's stale-screenshot rule: before
    acting on coordinates computed from `expected`, confirm the live
    page still matches it, so a page that mutated between perception and
    action doesn't get clicked using now-invalid coordinates.

    Deliberately does NOT call `capture_screenshot()`: that function
    also writes the PNG to disk and issues four separate `page.evaluate()`
    round-trips (devicePixelRatio, visualViewport.scale, scrollX,
    scrollY) to build a full `ScreenshotMeta` — none of which a pure
    hash comparison needs. Measured directly (not assumed): those four
    `page.evaluate()` calls cost ~3.1-3.4ms each and the disk write
    ~0.7ms, ~14ms of real, avoidable overhead added on top of the
    unavoidable ~90ms `page.screenshot()` call itself — and this
    function sits directly in the narrow gap between the staleness
    check and the actual `page.mouse.click()`/`keyboard.type()` call in
    `visipilot/action/runner.py` (implementation-plan.md B.10/C.1's
    documented "the gap is not literally zero" limitation): every
    millisecond spent here on work the comparison doesn't need widens
    that real race window for zero benefit. Hashing the raw screenshot
    bytes directly, with no disk I/O and no extra IPC round-trips,
    measurably narrows it instead (implementation-plan.md C.6).

    Uses `full_page=expected.meta.full_page` so the two images are
    directly comparable by dimensions, not confounded by capture mode.
    A plain hash compare, not a perceptual diff — verified empirically
    (not assumed) that repeated captures of an unchanged static page
    produce an identical hash in this project's headless-Chrome setup
    (no cursor-blink/antialiasing noise observed), so an exact mismatch
    reliably means real content changed, not capture jitter.
    """
    image_bytes = page.screenshot(full_page=expected.meta.full_page)
    return _hash_bytes(image_bytes) != expected.image_hash


def screenshot_changed_outside(old: Screenshot, new: Screenshot, exclude: BBox) -> bool:
    """Did anything change between `old` and `new` OUTSIDE `exclude`?

    Used in `visipilot/action/runner.py` right after a successful
    action, where `reference_screenshot` is rolled forward to `new` to
    absorb that action's own expected visual delta (most notably a TYPE
    step's echoed text) without the *next* step's staleness check seeing
    a false positive. Blindly trusting `new` as the fresh baseline is
    exactly what let an unrelated, real external change (e.g. a late
    banner) get silently absorbed too, since `new` reflects whatever the
    live page looked like at capture time, not just the acted-on
    element's own change — see implementation-plan.md C.7 for the traced
    root cause. Masking out `exclude` (the element the action just
    touched) before comparing isolates "did anything else on the page
    also change", which is exactly the question that must be answered
    before `new` is safe to trust as a baseline for elements `state`
    resolved from a completely different part of the page.

    Pixels-first, but deliberately NOT pixel-exact (unlike
    `screenshot_has_changed`) — measured directly against this project's
    own real test pages (not assumed), a pixel-exact compare here gives
    false positives: focusing and typing into an input measurably
    changes anti-aliasing/sub-pixel text rendering of *unrelated* text
    elsewhere on the page (e.g. a distant button's label switching from
    color-fringed sub-pixel rendering to grayscale rendering), a real
    Chromium rendering-mode side effect of the interaction itself, not a
    content change — confirmed by masking that noise out with a solid
    rectangle over `exclude` and still seeing the same diff appear at
    unrelated coordinates. `screenshot_has_changed`'s pixel-exact compare
    never hit this in practice only because the runner never previously
    compared two screenshots straddling the moment of an action — one
    always unconditionally overwrote the other.

    Masks `exclude`, then downsamples both images to a small fixed grid
    and compares with a magnitude threshold, which is coarse enough to
    average the above per-pixel text-rendering noise away while a real
    added/shifted element (a banner covering a meaningful area) still
    shows up clearly: five repeated real-browser measurements of the
    rendering-mode noise (typing into `search_basic.html`) peaked at a
    downsampled cell-diff of 63/765, completely stable across runs;
    five repeated measurements of a real late-appearing banner
    (`search_dynamic.html`) peaked at 337/765, equally stable — see
    implementation-plan.md C.7 for the full measurement. The threshold
    below sits with real margin on both sides of that measured gap.
    """
    old_image = Image.open(old.image_path).convert("RGB")
    new_image = Image.open(new.image_path).convert("RGB")
    if old_image.size != new_image.size:
        return True

    box = (
        max(0, int(exclude.x)),
        max(0, int(exclude.y)),
        min(old_image.width, int(exclude.x2) + 1),
        min(old_image.height, int(exclude.y2) + 1),
    )
    old_masked, new_masked = old_image.copy(), new_image.copy()
    ImageDraw.Draw(old_masked).rectangle(box, fill=(0, 0, 0))
    ImageDraw.Draw(new_masked).rectangle(box, fill=(0, 0, 0))

    old_thumb = old_masked.resize(_DIFF_THUMBNAIL_SIZE, Image.BILINEAR)
    new_thumb = new_masked.resize(_DIFF_THUMBNAIL_SIZE, Image.BILINEAR)
    old_arr = list(old_thumb.getdata())
    new_arr = list(new_thumb.getdata())
    max_cell_diff = max(
        abs(o[0] - n[0]) + abs(o[1] - n[1]) + abs(o[2] - n[2])
        for o, n in zip(old_arr, new_arr)
    )
    return max_cell_diff > _EXTERNAL_CHANGE_THRESHOLD
