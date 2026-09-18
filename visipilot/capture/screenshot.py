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

from playwright.sync_api import Page, sync_playwright

from visipilot.types import Screenshot, ScreenshotMeta

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent.parent / "out" / "screenshots"

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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

    Uses `full_page=expected.meta.full_page` so the two images are
    directly comparable by dimensions, not confounded by capture mode.
    A plain hash compare, not a perceptual diff — verified empirically
    (not assumed) that repeated captures of an unchanged static page
    produce an identical hash in this project's headless-Chrome setup
    (no cursor-blink/antialiasing noise observed), so an exact mismatch
    reliably means real content changed, not capture jitter.
    """
    fresh = capture_screenshot(page, full_page=expected.meta.full_page)
    return fresh.image_hash != expected.image_hash
