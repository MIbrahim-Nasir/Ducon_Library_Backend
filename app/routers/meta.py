"""Public metadata endpoints (build id, cache versioning)."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.build_meta import get_build_id

router = APIRouter(tags=["meta"])

_NO_STORE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
}


@router.get("/meta/build")
async def get_build_meta() -> JSONResponse:
    """Return the deploy build id so SPAs can flush stale PWA caches on landing."""
    build_id = get_build_id()
    return JSONResponse(
        content={"build_id": build_id, "cache_version": build_id},
        headers=_NO_STORE_HEADERS,
    )
