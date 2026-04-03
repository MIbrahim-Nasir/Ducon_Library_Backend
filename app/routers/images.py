import json
import os
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/images", tags=["images"])

METADATA_PATH = os.getenv("METADATA_PATH", "data/images/metadata.json")
CACHE_TTL = int(os.getenv("METADATA_CACHE_TTL", str(60 * 60 * 24)))  # default 1 day in seconds

_cache: list[dict] | None = None
_cache_loaded_at: float = 0.0


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _load_metadata() -> list[dict]:
    global _cache, _cache_loaded_at

    if _cache is not None and (time.time() - _cache_loaded_at) < CACHE_TTL:
        return _cache

    if _is_url(METADATA_PATH):
        response = httpx.get(METADATA_PATH, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch metadata from {METADATA_PATH} "
                f"(HTTP {response.status_code})"
            )
        data = response.json()
    else:
        local = Path(METADATA_PATH)
        if not local.exists():
            raise FileNotFoundError(f"metadata.json not found at {local}")
        with open(local, "r", encoding="utf-8") as f:
            data = json.load(f)

    _cache = data
    _cache_loaded_at = time.time()
    return _cache


@router.get("/metadata")
async def get_metadata():
    """
    Serves the image metadata.
    METADATA_PATH can be a local file path or a remote URL (e.g. Cloudflare R2 public URL).
    Response is cached in memory and refreshed at most once per METADATA_CACHE_TTL seconds (default 1 day).
    """
    try:
        return _load_metadata()
    except (FileNotFoundError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=str(e))
