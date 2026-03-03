import json
import os
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/images", tags=["images"])

METADATA_PATH = Path(
    os.getenv("METADATA_PATH", "data/images/metadata.json")
)
CACHE_TTL = int(os.getenv("METADATA_CACHE_TTL", str(60 * 60 * 24)))  # default 1 day in seconds

_cache: list[dict] | None = None
_cache_loaded_at: float = 0.0


def _load_metadata() -> list[dict]:
    global _cache, _cache_loaded_at

    if _cache is not None and (time.time() - _cache_loaded_at) < CACHE_TTL:
        return _cache

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"metadata.json not found at {METADATA_PATH}")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        _cache = json.load(f)
    _cache_loaded_at = time.time()
    return _cache


@router.get("/metadata")
async def get_metadata():
    """
    Serves the public image metadata file as-is.
    Cached in memory, reloaded from disk at most once per METADATA_CACHE_TTL seconds (default 1 day).
    """
    try:
        return _load_metadata()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
