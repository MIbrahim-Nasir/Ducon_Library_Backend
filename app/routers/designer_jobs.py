"""
Designer Agent job routes.

These routes turn the designer from a single chat turn into a background agent
run with progress events:

POST   /designer/jobs                  create and start a run
GET    /designer/jobs/{job_id}          inspect current state
GET    /designer/jobs/{job_id}/events   SSE progress stream
POST   /designer/jobs/{job_id}/cancel   cancel a running run
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse

from app.auth import get_current_user
from app.db.database import async_session_maker
from app.db.models import User
from app import storage
from app.designer_agent import (
    JOBS,
    cancel_job,
    create_job,
    get_job_for_user,
    run_designer_job,
    stream_job_events,
    stream_job_events_from_db,
)


router = APIRouter(prefix="/designer/jobs", tags=["designer-jobs"])

from app.sse import SSE_HEADERS as _SSE_HEADERS
from app.admin.settings_store import cfg_bool

def _live_debug() -> bool:
    return cfg_bool("LIVE_DEBUG", False)


def _dbg(*args) -> None:
    if _live_debug():
        print(*args)


@router.post("")
async def create_designer_job(
    request: Request,
    prompt: Optional[str] = Form(
        None,
        description="User design goal/suggestions. If omitted, agent designs independently.",
    ),
    model: str = Form("flash", description='"flash" or "pro" image generation model.'),
    aspect_ratio: Optional[str] = Form("16:9", description='Output aspect ratio, e.g. "16:9".'),
    user_image: UploadFile = File(..., description="Client space image to redesign."),
    current_user: User = Depends(get_current_user),
):
    """
    Start a long-running designer job.

    The frontend should upload the client's space image here, even if it also has
    a local IndexedDB upload id. Backend autonomous jobs need real image bytes.
    """
    raw = await user_image.read()
    if not raw:
        raise HTTPException(status_code=422, detail="user_image is empty.")

    if model not in {"flash", "pro"}:
        raise HTTPException(status_code=422, detail="model must be 'flash' or 'pro'.")

    embedding_model = request.app.state.embedding_model
    collection = request.app.state.collection

    job = create_job(current_user)
    _dbg(
        "[DESIGNER ROUTER ▶ CREATE]",
        {
            "job_id": job.id,
            "user_id": current_user.id,
            "prompt_chars": len(prompt or ""),
            "prompt_preview": (prompt or "")[:500],
            "model": model,
            "aspect_ratio": aspect_ratio,
            "upload_filename": user_image.filename,
            "upload_content_type": user_image.content_type,
            "upload_bytes": len(raw),
        },
    )

    async def _run():
        async with async_session_maker() as db:
            await run_designer_job(
                job=job,
                db=db,
                embedding_model=embedding_model,
                collection=collection,
                user_image_bytes=raw,
                user_prompt=prompt,
                model=model,
                aspect_ratio=aspect_ratio,
            )

    job.task = asyncio.create_task(_run(), name=f"designer-job-{job.id[:8]}")

    return {
        "job_id": job.id,
        "status": job.status,
        "events_url": f"/designer/jobs/{job.id}/events",
    }


@router.get("/{job_id}")
async def get_designer_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    job = await get_job_for_user(job_id, int(current_user.id))
    if job is None:
        raise HTTPException(status_code=404, detail="Designer job not found.")
    _dbg("[DESIGNER ROUTER ◀ GET]", {"job_id": job.id, "status": job.status, "events": len(job.events)})
    return {
        "job_id": job.id,
        "status": job.status,
        "created_at": job.created_at,
        "final": job.final,
        "error": job.error,
        "events": job.events,
    }


@router.get("/{job_id}/events")
async def designer_job_events(
    job_id: str,
    request: Request,
    token: Optional[str] = Query(default=None, description="JWT token for EventSource clients."),
):
    current_user = await _resolve_sse_user(request, token)
    uid = int(current_user.id)
    # Prefer the live in-memory queue when THIS worker owns the running job;
    # otherwise fall back to a DB-polling stream so cross-worker clients still
    # receive progress + a terminal event instead of a 404.
    live_job = JOBS.get(job_id)
    if live_job is not None and live_job.user_id == uid:
        _dbg("[DESIGNER ROUTER ▶ EVENTS]", {"job_id": live_job.id, "status": live_job.status, "existing_events": len(live_job.events), "mode": "live"})
        return StreamingResponse(
            stream_job_events(live_job),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    # Confirm ownership before streaming from the DB.
    owned = await get_job_for_user(job_id, uid)
    if owned is None:
        raise HTTPException(status_code=404, detail="Designer job not found.")
    _dbg("[DESIGNER ROUTER ▶ EVENTS]", {"job_id": job_id, "status": owned.status, "existing_events": len(owned.events), "mode": "db-poll"})
    return StreamingResponse(
        stream_job_events_from_db(job_id, uid),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/{job_id}/cancel")
async def cancel_designer_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    if not await cancel_job(job_id, int(current_user.id)):
        raise HTTPException(status_code=404, detail="Designer job not found.")
    _dbg("[DESIGNER ROUTER ▶ CANCEL]", {"job_id": job_id, "user_id": current_user.id})
    return {"job_id": job_id, "status": "cancelling"}


@router.get("/{job_id}/input-image")
async def get_designer_job_input_image(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Serve the user's ORIGINAL space photo for a designer job (the slider "before").

    The key is derived deterministically from (user_id, job_id) so no DB column is
    needed. Ownership is confirmed via the persisted job row (or the live in-memory
    job). The photo is stored WITHOUT a watermark — it is the client's own upload.

    Cloud mode: 302 redirect to a fresh presigned R2 URL.
    Local mode: serve the file directly via FileResponse.
    """
    # Confirm the job exists and belongs to this user before serving.
    job = await get_job_for_user(job_id, int(current_user.id))
    if job is None:
        raise HTTPException(status_code=404, detail="Designer job not found.")

    key = storage.designer_input_key(int(current_user.id), job_id)
    if storage.CLOUD_STORAGE:
        return RedirectResponse(
            url=storage.get_designer_input_url(job_id, key), status_code=302
        )
    path = storage.serve_designer_input_path(key)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Input image not found.")
    return FileResponse(path, media_type="image/png")


async def _resolve_sse_user(request: Request, token: Optional[str]) -> User:
    """
    Resolve user for SSE.

    Native EventSource cannot send Authorization headers, so this endpoint also
    accepts ?token=<jwt>. Fetch-based streaming can still use Bearer headers.
    Honors token revocation via ``resolve_user_from_token``.
    """
    raw_token = token
    if not raw_token:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            raw_token = auth.split(" ", 1)[1].strip()

    if not raw_token:
        raise HTTPException(status_code=401, detail="Authentication required.")

    async with async_session_maker() as db:
        from app.auth import resolve_user_from_token

        user = await resolve_user_from_token(raw_token, db)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return user
