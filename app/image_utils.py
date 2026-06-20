"""
Shared image loading and normalisation helpers.

Used by both the main autogenerate-images endpoint and the quotation router so
we avoid duplicating DB-fetch / resize logic across files.
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import HTTPException
from PIL import Image

# ── Constants ────────────────────────────────────────────────────────────────

MAX_IMAGE_PX = 2048

# ── Metadata cache ────────────────────────────────────────────────────────────

_IMAGE_INFO_JSON_PATH = os.getenv("IMAGE_INFO_JSON_PATH", "")
_IMAGE_INFO_CACHE: list[dict] | None = None
_IMAGE_INFO_CACHE_AT: float = 0.0
_IMAGE_INFO_CACHE_TTL = 60 * 60 * 24  # 1 day


def load_image_info() -> list[dict]:
    global _IMAGE_INFO_CACHE, _IMAGE_INFO_CACHE_AT
    now = time.time()
    if _IMAGE_INFO_CACHE is not None and (now - _IMAGE_INFO_CACHE_AT) < _IMAGE_INFO_CACHE_TTL:
        return _IMAGE_INFO_CACHE
    if not _IMAGE_INFO_JSON_PATH:
        return []
    if _IMAGE_INFO_JSON_PATH.startswith(("http://", "https://")):
        r = httpx.get(_IMAGE_INFO_JSON_PATH, timeout=30)
        data = r.json() if r.status_code == 200 else []
    else:
        p = Path(_IMAGE_INFO_JSON_PATH)
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    _IMAGE_INFO_CACHE = data
    _IMAGE_INFO_CACHE_AT = now
    return _IMAGE_INFO_CACHE


def get_image_metadata(filename: str) -> dict | None:
    """Return the metadata entry for a Ducon image by filename, or None."""
    try:
        return next(
            (item for item in load_image_info() if item.get("filename") == filename),
            None,
        )
    except Exception:
        return None


async def load_ducon_image(db_image) -> Image.Image:
    """
    Load a Ducon catalogue image as a PIL Image.
    Tries the local data/images/ folder first; falls back to the R2 URL.
    """
    local_path = Path(__file__).parent.parent / "data" / "images" / db_image.filename
    if local_path.exists():
        return Image.open(local_path)

    if not db_image.url:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Image '{db_image.filename}' not found locally "
                "and has no remote URL."
            ),
        )

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(db_image.url)

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Failed to fetch image '{db_image.filename}' from remote URL "
                f"(HTTP {response.status_code})."
            ),
        )

    return Image.open(io.BytesIO(response.content))


def normalize_user_image(image_bytes: bytes) -> Image.Image:
    """
    Normalise a user-uploaded image:
    - Converts HEIC/HEIF to RGB
    - Downsizes to MAX_IMAGE_PX on the longest side, preserving aspect ratio
    - Always returns RGB PIL Image
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > MAX_IMAGE_PX:
        scale = MAX_IMAGE_PX / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def mime_type(filename: Optional[str]) -> str:
    lower = (filename or "").lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"
