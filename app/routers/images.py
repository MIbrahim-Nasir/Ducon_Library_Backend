import json
import os
import time
import asyncio
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException

from app.admin.settings_store import cfg

router = APIRouter(prefix="/images", tags=["images"])

METADATA_PATH_DEFAULT = "data/images/metadata.json"
CACHE_TTL_DEFAULT = 60 * 60 * 24


def metadata_path() -> str:
    return cfg("METADATA_PATH", METADATA_PATH_DEFAULT)


def cache_ttl() -> int:
    return int(cfg("METADATA_CACHE_TTL", CACHE_TTL_DEFAULT))

_cache: list[dict] | None = None
_cache_loaded_at: float = 0.0


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _load_metadata() -> list[dict]:
    global _cache, _cache_loaded_at

    if _cache is not None and (time.time() - _cache_loaded_at) < cache_ttl():
        return _cache

    path = metadata_path()
    if _is_url(path):
        response = httpx.get(path, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch metadata from {path} "
                f"(HTTP {response.status_code})"
            )
        data = response.json()
    else:
        local = Path(path)
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
        # _load_metadata() may do a sync httpx.get (remote URL) + file read on a
        # cache miss — offload so the event loop keeps handling concurrent requests.
        return await asyncio.to_thread(_load_metadata)
    except (FileNotFoundError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail="Image metadata is temporarily unavailable.")
