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
from fastapi.responses import StreamingResponse
from jose import JWTError, jwt
from sqlalchemy import select

from app.auth import ALGORITHM, SECRET_KEY, get_current_user
from app.db.database import async_session_maker
from app.db.models import User
from app.designer_agent import (
    JOBS,
    cancel_job,
    create_job,
    run_designer_job,
    stream_job_events,
)


router = APIRouter(prefix="/designer/jobs", tags=["designer-jobs"])

from app.sse import SSE_HEADERS as _SSE_HEADERS
_LIVE_DEBUG: bool = os.getenv("LIVE_DEBUG", "").lower() in ("1", "true", "yes")


def _dbg(*args) -> None:
    if _LIVE_DEBUG:
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
    job = JOBS.get(job_id)
    if not job or job.user_id != current_user.id:
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
    job = JOBS.get(job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Designer job not found.")
    _dbg("[DESIGNER ROUTER ▶ EVENTS]", {"job_id": job.id, "status": job.status, "existing_events": len(job.events)})
    return StreamingResponse(
        stream_job_events(job),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/{job_id}/cancel")
async def cancel_designer_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    if not cancel_job(job_id, int(current_user.id)):
        raise HTTPException(status_code=404, detail="Designer job not found.")
    _dbg("[DESIGNER ROUTER ▶ CANCEL]", {"job_id": job_id, "user_id": current_user.id})
    return {"job_id": job_id, "status": "cancelling"}


async def _resolve_sse_user(request: Request, token: Optional[str]) -> User:
    """
    Resolve user for SSE.

    Native EventSource cannot send Authorization headers, so this endpoint also
    accepts ?token=<jwt>. Fetch-based streaming can still use Bearer headers.
    """
    raw_token = token
    if not raw_token:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            raw_token = auth.split(" ", 1)[1].strip()

    if not raw_token:
        raise HTTPException(status_code=401, detail="Authentication required.")

    try:
        payload = jwt.decode(raw_token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    async with async_session_maker() as db:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return user
