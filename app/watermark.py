"""
app/watermark.py
================
PIL-based post-processing watermark applied to every generated image BEFORE it
is uploaded to Cloudflare R2 / saved to local disk.

Two marks are composited onto each image:

1. **Main watermark** (``ducon_main_watermark.png``) — large Ducon "D" letter,
   scaled to fit entirely within the image (no cut-off), centered, at
   configurable opacity.

2. **Corner watermark** (``ducon_corner_watermark.png``) — contact/logo badge
   placed flush to the bottom-right corner (edges align with the image edges;
   no margin), always at **100% opacity** (ignores ``WATERMARK_OPACITY``).

Assets are expected to be RGBA PNGs with real transparency. If an asset has a
solid near-black background (legacy JPEG / flattened export), that background
is treated as transparent before compositing.

Configuration is read live via ``cfg()`` from ``app.admin.settings_store`` so
admin changes take effect without a restart.

Public API:
    apply_watermark(image: Image.Image) -> Image.Image
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from PIL import Image

from app.admin.settings_store import cfg, cfg_bool, cfg_str

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────

_DEFAULT_MAIN_WATERMARK_PATH = os.path.join("app", "static", "ducon_main_watermark.png")
_DEFAULT_CORNER_WATERMARK_PATH = os.path.join("app", "static", "ducon_corner_watermark.png")

# Main mark covers most of the shorter side while staying fully inside bounds.
_MAIN_FIT_RATIO = 0.92

# Corner badge: target width as a fraction of image width (capped by height).
# ~20% larger than the original 0.22 / 0.12 ratios (noticeable, not dominant).
_CORNER_WIDTH_RATIO = 0.26
_CORNER_MAX_HEIGHT_RATIO = 0.14

# Near-black RGB treated as transparent when the asset lacks a real alpha channel
# (or has an opaque black matte).
_BLACK_KEY_THRESHOLD = 18

_logo_cache: dict[tuple[str, float], Image.Image] = {}


# ── Loading ──────────────────────────────────────────────────────────────────

def _configured_main_watermark_path() -> str:
    return cfg_str("WATERMARK_MAIN_PATH", _DEFAULT_MAIN_WATERMARK_PATH)


def _configured_corner_watermark_path() -> str:
    return cfg_str("WATERMARK_CORNER_PATH", _DEFAULT_CORNER_WATERMARK_PATH)


def _black_to_alpha(img: Image.Image, threshold: int = _BLACK_KEY_THRESHOLD) -> Image.Image:
    """Treat near-black pixels as transparent; preserve existing alpha otherwise."""
    rgba = img.convert("RGBA")
    pixels = rgba.load()
    w, h = rgba.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            if r <= threshold and g <= threshold and b <= threshold:
                pixels[x, y] = (r, g, b, 0)
            # else keep original alpha
    return rgba


def _load_watermark_asset(path: str, *, key_black: bool = True) -> Optional[Image.Image]:
    """Load and cache a watermark PNG/JPEG as RGBA."""
    if not path or not os.path.isfile(path):
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None

    cached = _logo_cache.get((path, mtime))
    if cached is not None:
        return cached.copy()

    try:
        raw = Image.open(path)
        has_alpha = "A" in raw.getbands()
        rgba = raw.convert("RGBA")
        # Key near-black only for opaque mattes (JPEG / flattened exports).
        # Real RGBA assets with transparency are left untouched.
        if key_black and (not has_alpha or _has_solid_black_matte(rgba)):
            rgba = _black_to_alpha(rgba)
    except Exception as exc:
        logger.warning("[watermark] Failed to load asset %s: %s", path, exc)
        return None

    _logo_cache[(path, mtime)] = rgba
    return rgba.copy()


def _has_solid_black_matte(rgba: Image.Image, sample_step: int = 16) -> bool:
    """Heuristic: many near-black fully-opaque pixels suggest a black matte."""
    w, h = rgba.size
    black_opaque = 0
    total = 0
    px = rgba.load()
    for y in range(0, h, sample_step):
        for x in range(0, w, sample_step):
            r, g, b, a = px[x, y]
            total += 1
            if a >= 250 and r <= _BLACK_KEY_THRESHOLD and g <= _BLACK_KEY_THRESHOLD and b <= _BLACK_KEY_THRESHOLD:
                black_opaque += 1
    return total > 0 and (black_opaque / total) >= 0.15


def _scale_to_fit(mark: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Scale mark to fit inside (max_w, max_h) without cropping (contain)."""
    mw, mh = mark.size
    if mw <= 0 or mh <= 0 or max_w <= 0 or max_h <= 0:
        return mark
    scale = min(max_w / mw, max_h / mh)
    if scale <= 0:
        return mark
    new_size = (max(1, int(mw * scale)), max(1, int(mh * scale)))
    if new_size == mark.size:
        return mark
    return mark.resize(new_size, Image.LANCZOS)


def _apply_opacity(mark: Image.Image, opacity: float) -> Image.Image:
    """Scale the alpha channel of an RGBA image by opacity (0..1)."""
    alpha_factor = max(0.0, min(1.0, opacity))
    if alpha_factor >= 1.0:
        return mark
    out = mark.copy()
    alpha = out.getchannel("A")
    alpha = alpha.point(lambda a: int(a * alpha_factor))
    out.putalpha(alpha)
    return out


# ── Mark rendering ───────────────────────────────────────────────────────────

def _render_main_watermark(width: int, height: int, opacity: float) -> Image.Image:
    """Centered main 'D' watermark, scaled to fit fully inside the image."""
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    mark = _load_watermark_asset(_configured_main_watermark_path(), key_black=True)
    if mark is None:
        return overlay

    max_w = max(1, int(width * _MAIN_FIT_RATIO))
    max_h = max(1, int(height * _MAIN_FIT_RATIO))
    fitted = _scale_to_fit(mark, max_w, max_h)
    fitted = _apply_opacity(fitted, opacity)

    mw, mh = fitted.size
    x = (width - mw) // 2
    y = (height - mh) // 2
    overlay.alpha_composite(fitted, (x, y))
    return overlay


def _render_corner_watermark(width: int, height: int) -> Image.Image:
    """Bottom-right corner badge, flush to image edges (no padding), fully opaque.

    Does not apply ``WATERMARK_OPACITY`` — the corner mark always composites at
    the asset's native alpha (100% opacity relative to admin opacity settings).
    """
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    corner = _load_watermark_asset(_configured_corner_watermark_path(), key_black=True)
    if corner is None:
        return overlay

    # Drop transparent padding so flush-right aligns to visible content.
    bbox = corner.getbbox()
    if bbox:
        corner = corner.crop(bbox)

    min_dim = min(width, height)
    target_w = max(1, int(width * _CORNER_WIDTH_RATIO))
    scale = target_w / corner.width
    target_h = max(1, int(corner.height * scale))
    max_h = max(1, int(min_dim * _CORNER_MAX_HEIGHT_RATIO))
    if target_h > max_h:
        scale = max_h / corner.height
        target_w = max(1, int(corner.width * scale))
        target_h = max_h

    # Full opacity: never pass through WATERMARK_OPACITY / _apply_opacity.
    badge = corner.resize((target_w, target_h), Image.LANCZOS)
    # Flush bottom-right: x=width-badge_w, y=height-badge_h
    x = max(0, width - target_w)
    y = max(0, height - target_h)
    overlay.alpha_composite(badge, (x, y))
    return overlay


def _watermark_enabled() -> bool:
    """True when watermarking is on. ``ENABLE_WATERMARK`` env wins when set."""
    raw = os.getenv("ENABLE_WATERMARK")
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return cfg_bool("WATERMARK_ENABLED", True)


# ── Public API ───────────────────────────────────────────────────────────────

def apply_watermark(image: Image.Image) -> Image.Image:
    """Apply centered main + bottom-right corner watermarks; return RGB image.

    No-op (returns an RGB copy) when watermarking is disabled. Rendering errors
    never propagate — the unmarked image is returned so a glitch cannot block
    a generation from being saved.
    """
    try:
        if not _watermark_enabled():
            return image.convert("RGB")

        opacity = float(cfg("WATERMARK_OPACITY", 0.4))

        base = image.convert("RGBA")
        width, height = base.size

        main = _render_main_watermark(width, height, opacity)
        base = Image.alpha_composite(base, main)

        corner = _render_corner_watermark(width, height)
        base = Image.alpha_composite(base, corner)

        return base.convert("RGB")
    except Exception as exc:
        logger.warning("[watermark] apply_watermark failed, returning unmarked image: %s", exc)
        try:
            return image.convert("RGB")
        except Exception:
            return image
