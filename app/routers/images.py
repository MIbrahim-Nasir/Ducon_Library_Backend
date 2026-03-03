import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/images", tags=["images"])

# Path to the metadata JSON file maintained by the admin.
# This file already contains full image URLs pointing to the VPS.
METADATA_PATH = Path(
    os.getenv("METADATA_PATH", "data/images/metadata.json")
)


@lru_cache(maxsize=1)
def _load_metadata() -> list[dict]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"metadata.json not found at {METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/metadata")
async def get_metadata():
    """
    Serves the public image metadata file as-is.
    Called once when the frontend loads.
    Returns the full list with all fields and image URLs as built by the admin.
    """
    try:
        return _load_metadata()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
