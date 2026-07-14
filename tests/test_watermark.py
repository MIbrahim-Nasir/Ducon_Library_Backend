"""Placement tests for centered main + bottom-right corner watermarks."""
from __future__ import annotations

import os

import pytest
from PIL import Image

from app import watermark as wm


@pytest.fixture(autouse=True)
def _clear_logo_cache():
    wm._logo_cache.clear()
    yield
    wm._logo_cache.clear()


@pytest.fixture
def enable_watermark(monkeypatch):
    monkeypatch.setenv("ENABLE_WATERMARK", "true")
    monkeypatch.setattr(wm, "cfg", lambda key, default=None: {
        "WATERMARK_OPACITY": 1.0,
        "WATERMARK_MAIN_PATH": "app/static/ducon_main_watermark.png",
        "WATERMARK_CORNER_PATH": "app/static/ducon_corner_watermark.png",
    }.get(key, default))
    monkeypatch.setattr(wm, "cfg_str", lambda key, default="": {
        "WATERMARK_MAIN_PATH": "app/static/ducon_main_watermark.png",
        "WATERMARK_CORNER_PATH": "app/static/ducon_corner_watermark.png",
    }.get(key, default))
    monkeypatch.setattr(wm, "cfg_bool", lambda key, default=False: True)


def test_watermark_assets_exist():
    assert os.path.isfile("app/static/ducon_main_watermark.png")
    assert os.path.isfile("app/static/ducon_corner_watermark.png")


def test_main_watermark_fits_and_is_centered(enable_watermark):
    base = Image.new("RGB", (800, 600), color=(180, 180, 180))
    marked = wm.apply_watermark(base)
    assert marked.size == (800, 600)
    assert marked.mode == "RGB"

    overlay = wm._render_main_watermark(800, 600, opacity=1.0)
    bbox = overlay.getbbox()
    assert bbox is not None
    left, top, right, bottom = bbox
    # Fully inside image bounds (no cut-off).
    assert left >= 0 and top >= 0 and right <= 800 and bottom <= 600
    # Roughly centered: mark center near image center.
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    assert abs(cx - 400) < 40
    assert abs(cy - 300) < 40
    # Large: covers a substantial portion of the shorter side.
    mark_h = bottom - top
    mark_w = right - left
    assert mark_h >= 600 * 0.5
    assert mark_w >= 600 * 0.4


def test_corner_watermark_flush_bottom_right(enable_watermark):
    base = Image.new("RGB", (1000, 800), color=(200, 200, 200))
    overlay = wm._render_corner_watermark(1000, 800)
    bbox = overlay.getbbox()
    assert bbox is not None
    left, top, right, bottom = bbox
    # Flush to right and bottom edges (allow 1px resampling slack).
    assert right >= 999
    assert bottom >= 799
    # Scale ~_CORNER_WIDTH_RATIO (0.26) / _CORNER_MAX_HEIGHT_RATIO (0.14).
    assert left > 1000 * 0.60
    assert (bottom - top) <= 800 * 0.14 + 2
    # Noticeably larger than the old 0.22 / 0.12 ratios, but not dominant.
    assert (right - left) >= 1000 * 0.18
    assert (bottom - top) >= min(1000, 800) * 0.08


def test_corner_watermark_ignores_main_opacity(enable_watermark, monkeypatch):
    """Corner stays fully opaque even when WATERMARK_OPACITY is low."""
    monkeypatch.setattr(wm, "cfg", lambda key, default=None: {
        "WATERMARK_OPACITY": 0.1,
        "WATERMARK_MAIN_PATH": "app/static/ducon_main_watermark.png",
        "WATERMARK_CORNER_PATH": "app/static/ducon_corner_watermark.png",
    }.get(key, default))

    overlay = wm._render_corner_watermark(1000, 800)
    # Corner path never calls _apply_opacity; solid pixels keep full alpha.
    # (Anti-aliased edges may be <255 — only require that some pixels are fully opaque.)
    alphas = [
        overlay.getpixel((x, y))[3]
        for y in range(800 - 1, max(0, 800 - 120), -1)
        for x in range(max(0, 1000 - 300), 1000)
    ]
    assert max(alphas) == 255
    # If WATERMARK_OPACITY (0.1) had been applied, max alpha would be ~25.
    assert max(alphas) > 50


def test_apply_watermark_composites_both(enable_watermark):
    base = Image.new("RGB", (640, 480), color=(120, 140, 160))
    marked = wm.apply_watermark(base)
    # Marked image must differ from the flat base somewhere.
    assert marked.getpixel((320, 240)) != base.getpixel((320, 240)) or marked.getpixel((639, 479)) != base.getpixel((639, 479))


def test_watermark_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("ENABLE_WATERMARK", "false")
    base = Image.new("RGB", (100, 100), color=(10, 20, 30))
    marked = wm.apply_watermark(base)
    assert marked.size == (100, 100)
    assert marked.getpixel((50, 50)) == (10, 20, 30)


def test_black_to_alpha_keys_matte():
    img = Image.new("RGBA", (4, 4), (0, 0, 0, 255))
    img.putpixel((1, 1), (255, 0, 0, 255))
    out = wm._black_to_alpha(img, threshold=18)
    assert out.getpixel((0, 0))[3] == 0
    assert out.getpixel((1, 1)) == (255, 0, 0, 255)


def test_scale_to_fit_no_cutoff():
    mark = Image.new("RGBA", (400, 200), (255, 255, 255, 128))
    fitted = wm._scale_to_fit(mark, 100, 100)
    assert fitted.width <= 100
    assert fitted.height <= 100
    assert fitted.width == 100  # limited by width for 2:1 mark in square box
    assert fitted.height == 50
