"""
app/watermark.py
================
PIL-based post-processing watermark applied to every generated image BEFORE it
is uploaded to Cloudflare R2 / saved to local disk.

Two marks are composited onto each image:

1. A large translucent diagonal watermark — the Ducon logo (or "DUCON" text if
   no usable logo asset exists) tiled along a diagonal across the middle of
   the image, rotated ~32°, at ~40% opacity (configurable). Visible but not
   obstructive.

2. A small opaque corner badge — the precomposed ``corner watermark.jpeg``
   (Ducon logo + phone numbers) placed top-left at a modest size.

Adaptive logo selection
-----------------------
Two logo variants are shipped: a white logo (for dark image regions) and a
black logo (for light image regions). For each image the average luminance is
computed by downscaling to 64x64 grayscale and taking the mean pixel value.
If mean luminance < 128 → the WHITE logo is used (visible on dark). Otherwise
the BLACK logo is used (visible on light). The corner badge text color and the
rounded-rectangle backdrop follow the same adaptive choice so the badge stays
legible on any background.

Both marks are rendered with Pillow (paste with alpha, ImageDraw/ImageFont,
.rotate with expand). Nothing here touches the AI generation prompt.

Configuration is read live via ``cfg()`` from ``app.admin.settings_store`` so
admin changes (enable/disable, opacity, logo paths) take effect immediately
without a restart.

Public API:
    apply_watermark(image: Image.Image) -> Image.Image
        Returns a new RGB image with the watermark applied. No-op (returns a
        copy of the input as RGB) when ENABLE_WATERMARK / WATERMARK_ENABLED is false.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from PIL import Image, ImageDraw, ImageFont, ImageStat

from app.admin.settings_store import cfg, cfg_bool, cfg_str

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_DIAGONAL_TEXT = "DUCON"

# Rotation angle (degrees) for the diagonal mark.
_DIAGONAL_ANGLE = 32

# Luminance threshold for adaptive logo selection. Mean luminance below this
# → image is "dark" → white logo. >= this → "light" → black logo.
_BRIGHTNESS_THRESHOLD = 128

# Downscale size used to estimate average luminance cheaply.
_LUMA_SAMPLE_SIZE = 64

# Default logo paths (relative to repo root). Admins can override via the
# WATERMARK_LOGO_* settings and/or drop PNGs at these locations.
_DEFAULT_LOGO_PATH = os.path.join("app", "static", "ducon_logo_white.png")
_DEFAULT_LOGO_WHITE_PATH = os.path.join("app", "static", "ducon_logo_white.png")
_DEFAULT_LOGO_BLACK_PATH = os.path.join("app", "static", "ducon_logo_black.png")
_DEFAULT_CORNER_WATERMARK_PATH = os.path.join("app", "static", "corner watermark.jpeg")

# Diagonal mark size as a fraction of the image's shorter side.
_DIAGONAL_SIZE_RATIO = 0.52

# Corner JPEG: target width as a fraction of image width (capped by height).
_CORNER_WIDTH_RATIO = 0.20
_CORNER_MAX_HEIGHT_RATIO = 0.09

# Optional per-color opacity boost. White-on-dark tends to look fainter than
# black-on-light at the same alpha, so we nudge it up a touch. WATERMARK_OPACITY
# remains the user-facing knob; this is an internal tuning factor.
_OPACITY_BOOST_WHITE = 1.15
_OPACITY_BOOST_BLACK = 1.0

# Windows + cross-platform font candidates for the bold "DUCON" mark and the
# phone-number text. Tried in order; load_default is the final fallback.
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\arialbd.ttf",   # Arial Bold (Windows)
    r"C:\Windows\Fonts\ariblk.ttf",    # Arial Black (Windows)
    r"C:\Windows\Fonts\segoeuib.ttf",  # Segoe UI Bold (Windows)
    r"C:\Windows\Fonts\segoeui.ttf",   # Segoe UI (Windows)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",       # Linux
    "/Library/Fonts/Arial Bold.ttf",   # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",     # macOS
)

# Cache for the raw loaded logos so we don't hit disk on every call. Keyed by
# (path, mtime) so admin path changes / file swaps are picked up.
_logo_cache: dict[tuple[str, float], Image.Image] = {}


# ── Font helpers ─────────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.ImageFont:
    """Return a TrueType font at ``size`` px, falling back to PIL's default."""
    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except (OSError, IOError):
            continue
    # Final fallback: PIL's bitmap default font (tiny; callers scale the render).
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    """Return (width, height) of ``text`` drawn with ``font``."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# ── Logo loading ─────────────────────────────────────────────────────────────

def _configured_logo_path() -> str:
    return str(cfg("WATERMARK_LOGO_PATH", _DEFAULT_LOGO_PATH))


def _configured_white_logo_path() -> str:
    return str(cfg("WATERMARK_LOGO_WHITE_PATH", _DEFAULT_LOGO_WHITE_PATH))


def _configured_black_logo_path() -> str:
    return str(cfg("WATERMARK_LOGO_BLACK_PATH", _DEFAULT_LOGO_BLACK_PATH))


def _load_logo_from(path: str) -> Optional[Image.Image]:
    """Load and cache a single logo PNG (RGBA) from ``path``.

    Returns None if the file is missing / fails to load.
    """
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
        logo = Image.open(path).convert("RGBA")
    except Exception as exc:  # corrupt / unsupported file
        logger.warning("[watermark] Failed to load logo %s: %s", path, exc)
        return None

    _logo_cache[(path, mtime)] = logo
    return logo.copy()


def _load_raw_logo() -> Optional[Image.Image]:
    """Backward-compat single-logo loader (WATERMARK_LOGO_PATH)."""
    return _load_logo_from(_configured_logo_path())


def _configured_logo_path_is_override() -> bool:
    """True when WATERMARK_LOGO_PATH points to a custom file (single-logo override).

    The default value equals the white logo path, so by default the adaptive
    white/black selection is used. If an admin points WATERMARK_LOGO_PATH at a
    different existing file, that single logo overrides both adaptive variants.
    """
    path = _configured_logo_path()
    if not path or not os.path.isfile(path):
        return False
    default_white = os.path.normpath(_DEFAULT_LOGO_WHITE_PATH)
    default_black = os.path.normpath(_DEFAULT_LOGO_BLACK_PATH)
    norm = os.path.normpath(path)
    return norm != default_white and norm != default_black


def _load_white_logo() -> Optional[Image.Image]:
    return _load_logo_from(_configured_white_logo_path())


def _load_black_logo() -> Optional[Image.Image]:
    return _load_logo_from(_configured_black_logo_path())


def _select_logo(is_dark: bool) -> tuple[Optional[Image.Image], str]:
    """Pick the adaptively appropriate logo, with fallbacks.

    Single-logo override: if WATERMARK_LOGO_PATH points at a custom existing
    file (different from the two adaptive defaults), that single logo is used
    for both colors and its own brightness decides the text/rectangle color tag.

    Adaptive order:
      - dark image → white logo, falling back to black logo, then None.
      - light image → black logo, falling back to white logo, then None.

    Returns (logo_or_None, color_tag) where color_tag is "white" | "black" |
    "none" describing which variant was actually loaded (used by callers to
    pick text/rectangle colors).
    """
    if _configured_logo_path_is_override():
        logo = _load_raw_logo()
        if logo is not None:
            # Derive color tag from the logo's own average luminance.
            try:
                luma = float(ImageStat.Stat(logo.convert("L")).mean[0])
            except Exception:
                luma = _BRIGHTNESS_THRESHOLD
            return logo, "white" if luma >= _BRIGHTNESS_THRESHOLD else "black"

    if is_dark:
        logo = _load_white_logo()
        if logo is not None:
            return logo, "white"
        logo = _load_black_logo()
        if logo is not None:
            return logo, "black"
        return None, "none"
    else:
        logo = _load_black_logo()
        if logo is not None:
            return logo, "black"
        logo = _load_white_logo()
        if logo is not None:
            return logo, "white"
        return None, "none"


def _scaled_logo_from(logo: Image.Image, target_px: int) -> Optional[Image.Image]:
    """Logo scaled so its largest dimension ≈ target_px (preserving aspect)."""
    w, h = logo.size
    if w == 0 or h == 0:
        return None
    scale = target_px / max(w, h)
    if scale <= 0:
        return None
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return logo.resize(new_size, Image.LANCZOS)


# ── Brightness ───────────────────────────────────────────────────────────────

def _mean_luminance(image: Image.Image) -> float:
    """Cheap average luminance estimate (0-255). Downscale to 64x64 L mode."""
    try:
        small = image.convert("L").resize((_LUMA_SAMPLE_SIZE, _LUMA_SAMPLE_SIZE))
        return float(ImageStat.Stat(small).mean[0])
    except Exception:
        # Fall back to "neutral" brightness → pick white logo (safe default).
        return float(_BRIGHTNESS_THRESHOLD - 1)


# ── Mark rendering ───────────────────────────────────────────────────────────

def _render_text_mark(text: str, target_px: int, color: tuple[int, int, int, int] = (255, 255, 255, 255)) -> Image.Image:
    """Render ``text`` as a bold mark sized to ~target_px tall (RGBA)."""
    font_size = max(8, int(target_px))
    font = _load_font(font_size)

    # Measure on a scratch draw to size the canvas precisely.
    scratch = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(scratch)
    tw, th = _text_size(draw, text, font=font)

    # If we fell back to load_default (tiny bitmap), th will be ~11px regardless
    # of font_size. Render at the default size then scale up to target_px.
    if th <= 0 or (th < target_px * 0.5 and font_size > 24):
        pad = 4
        canvas = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
        d2 = ImageDraw.Draw(canvas)
        d2.text((pad, pad), text, fill=color, font=font)
        scale = target_px / canvas.height
        canvas = canvas.resize(
            (max(1, int(canvas.width * scale)), max(1, int(canvas.height * scale))),
            Image.LANCZOS,
        )
        return canvas

    pad = max(4, int(target_px * 0.08))
    canvas = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((pad, pad), text, fill=color, font=font)
    return canvas


def _render_diagonal_overlay(
    width: int,
    height: int,
    opacity: float,
    logo: Optional[Image.Image],
    color_tag: str,
) -> Image.Image:
    """Build the full-image translucent diagonal watermark overlay (RGBA)."""
    min_dim = min(width, height)
    mark_target = int(min_dim * _DIAGONAL_SIZE_RATIO)

    if logo is not None:
        base_mark = _scaled_logo_from(logo, mark_target)
    else:
        # Text fallback color follows adaptive choice: white on dark, black on light.
        text_color = (255, 255, 255, 255) if color_tag == "white" else (0, 0, 0, 255)
        base_mark = _render_text_mark(_DIAGONAL_TEXT, mark_target, color=text_color)

    if base_mark is None:
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # Rotate the mark (expand so corners aren't clipped). Transparent canvas →
    # rotate keeps alpha.
    rotated = base_mark.rotate(_DIAGONAL_ANGLE, expand=True, resample=Image.BICUBIC)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # Tile the rotated mark along the main diagonal so a single mark doesn't
    # look sparse on wide/tall images. Step size based on mark size.
    mw, mh = rotated.size
    diag_len = int((width ** 2 + height ** 2) ** 0.5)
    step = max(int(max(mw, mh) * 1.6), 1)

    # Walk along the diagonal from bottom-left toward top-right, centering the
    # line on the image. We paste marks whose centers lie on the diagonal.
    n = max(1, diag_len // step)
    for i in range(-1, n + 2):
        t = (i + 1) / (n + 1)  # 0..1 along the diagonal
        cx = int(t * width)
        cy = int((1 - t) * height)
        # Skip marks whose center is well outside the image (keep edge ones so
        # the diagonal reads continuously).
        if cx < -mw or cx > width + mw or cy < -mh or cy > height + mh:
            continue
        pos = (cx - mw // 2, cy - mh // 2)
        overlay.alpha_composite(rotated, pos)

    # Apply the configured opacity (with optional per-color boost) by scaling
    # the whole overlay's alpha channel.
    boost = _OPACITY_BOOST_WHITE if color_tag == "white" else _OPACITY_BOOST_BLACK
    alpha_factor = max(0.0, min(1.0, opacity * boost))
    if alpha_factor < 1.0:
        alpha = overlay.getchannel("A")
        alpha = alpha.point(lambda a: int(a * alpha_factor))
        overlay.putalpha(alpha)

    return overlay


def _configured_corner_watermark_path() -> str:
    return cfg_str("WATERMARK_CORNER_PATH", _DEFAULT_CORNER_WATERMARK_PATH)


def _render_corner_watermark(width: int, height: int) -> Image.Image:
    """Build the small opaque top-left corner badge from corner watermark.jpeg."""
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    corner = _load_logo_from(_configured_corner_watermark_path())
    if corner is None:
        return overlay

    min_dim = min(width, height)
    target_w = max(1, int(width * _CORNER_WIDTH_RATIO))
    scale = target_w / corner.width
    target_h = max(1, int(corner.height * scale))
    max_h = max(1, int(min_dim * _CORNER_MAX_HEIGHT_RATIO))
    if target_h > max_h:
        scale = max_h / corner.height
        target_w = max(1, int(corner.width * scale))
        target_h = max_h

    badge = corner.resize((target_w, target_h), Image.LANCZOS)
    margin = max(8, int(min_dim * 0.02))
    overlay.alpha_composite(badge, (margin, margin))
    return overlay


def _watermark_enabled() -> bool:
    """True when watermarking is on. ``ENABLE_WATERMARK`` env wins when set."""
    raw = os.getenv("ENABLE_WATERMARK")
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return cfg_bool("WATERMARK_ENABLED", True)


# ── Public API ───────────────────────────────────────────────────────────────

def apply_watermark(image: Image.Image) -> Image.Image:
    """Apply the diagonal + corner watermark and return a new RGB image.

    No-op (returns an RGB copy) when ``ENABLE_WATERMARK`` / ``WATERMARK_ENABLED``
    is false. Errors in
    watermark rendering never propagate — the original image is returned so a
    rendering glitch can never block a generation from being saved.
    """
    try:
        if not _watermark_enabled():
            return image.convert("RGB")

        opacity = float(cfg("WATERMARK_OPACITY", 0.4))

        base = image.convert("RGBA")
        width, height = base.size

        # ── Adaptive logo selection based on image brightness ─────────────────
        mean_luma = _mean_luminance(base)
        is_dark = mean_luma < _BRIGHTNESS_THRESHOLD
        logo, color_tag = _select_logo(is_dark)
        logger.debug(
            "[watermark] mean_luma=%.1f is_dark=%s color_tag=%s logo_loaded=%s",
            mean_luma, is_dark, color_tag, logo is not None,
        )

        diagonal = _render_diagonal_overlay(width, height, opacity, logo, color_tag)
        base = Image.alpha_composite(base, diagonal)

        corner = _render_corner_watermark(width, height)
        base = Image.alpha_composite(base, corner)

        return base.convert("RGB")
    except Exception as exc:
        logger.warning("[watermark] apply_watermark failed, returning unmarked image: %s", exc)
        try:
            return image.convert("RGB")
        except Exception:
            return image
