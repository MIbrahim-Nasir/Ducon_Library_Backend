"""
Generation job routes — create + poll/SSE without one long SSE request.

POST   /generation/jobs                  create and start a run
GET    /generation/jobs/{job_id}          inspect current state
GET    /generation/jobs/{job_id}/events   SSE progress stream
POST   /generation/jobs/{job_id}/cancel   cancel a running run

The existing POST /generate-multi-image SSE endpoint remains for clients that
still stream in one request.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_optional_user
from app.db.database import get_db
from app.db.models import GuestSession, User
from app.generation_jobs import (
    JOBS,
    cancel_job,
    create_job,
    get_job_for_owner,
    persist_job_create,
    run_generation_job,
    stream_job_events,
    stream_job_events_from_db,
)
from app.middleware.request_id import get_request_id
from app.routers.guest import resolve_guest_context
from app.routers.multi_image_gen import parse_multi_image_request
from app.sse import SSE_HEADERS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generation/jobs", tags=["generation-jobs"])


async def _resolve_owner(
    request: Request,
    current_user: User | None,
    db: AsyncSession,
    *,
    cf_turnstile_token: Optional[str],
) -> tuple[Optional[int], Optional[str], Optional[GuestSession]]:
    if current_user is not None:
        return int(current_user.id), None, None

    guest_row = await resolve_guest_context(
        request, db,
        turnstile_token=cf_turnstile_token,
        endpoint="/generation/jobs",
        source="generation_jobs",
    )
    await db.commit()
    return None, guest_row.session_id, guest_row


async def _resolve_sse_owner(
    request: Request,
    token: Optional[str],
) -> tuple[Optional[int], Optional[str]]:
    """Auth for EventSource (?token=) or Authorization / guest cookie+header.

    JWT path goes through ``resolve_user_from_token`` so revocation is honored.
    """
    raw_token = token
    if not raw_token:
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            raw_token = auth.split(" ", 1)[1].strip()

    if raw_token:
        from app.auth import resolve_user_from_token
        from app.db.database import async_session_maker

        async with async_session_maker() as db:
            user = await resolve_user_from_token(raw_token, db)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return int(user.id), None

    from app.guest_session_token import resolve_guest_session_id

    guest_id = resolve_guest_session_id(request)
    if guest_id:
        return None, guest_id

    raise HTTPException(status_code=401, detail="Authentication required")


@router.post("")
@router.post("/")
async def create_generation_job(
    request: Request,
    prompt: str = Form(..., description="Full task prompt for the generation model."),
    images_meta: Optional[str] = Form(None),
    image_specs: Optional[str] = Form(None),
    model: str = Form("pro"),
    aspect_ratio: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
    cf_turnstile_token: Optional[str] = Form(None),
    current_user: User | None = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    user_id, guest_session_id, _guest_row = await _resolve_owner(
        request, current_user, db, cf_turnstile_token=cf_turnstile_token
    )

    descriptors = await parse_multi_image_request(
        prompt=prompt,
        model=model,
        aspect_ratio=aspect_ratio,
        images_meta=images_meta,
        image_specs=image_specs,
        files=files,
        log_tag="GenerationJobs",
    )

    job = create_job(
        user_id=user_id,
        guest_session_id=guest_session_id,
        request_id=get_request_id(),
    )

    # Persist BEFORE returning the job_id: with multiple workers the client's
    # immediate GET /events usually lands on a different worker, whose only
    # lookup path is this row. Persisting inside the background task raced it.
    if not await persist_job_create(job):
        JOBS.pop(job.id, None)
        raise HTTPException(
            status_code=503,
            detail="Could not start generation job. Please try again.",
        )

    async def _run():
        await run_generation_job(
            job=job,
            prompt=prompt,
            descriptors=descriptors,
            model=model,
            aspect_ratio=aspect_ratio,
        )

    job.task = asyncio.create_task(_run(), name=f"generation-job-{job.id[:8]}")

    return {
        "job_id": job.id,
        "status": job.status,
        "events_url": f"/generation/jobs/{job.id}/events",
        "request_id": job.request_id,
    }


@router.get("/{job_id}")
async def get_generation_job(
    job_id: str,
    request: Request,
    current_user: User | None = Depends(get_optional_user),
):
    user_id = int(current_user.id) if current_user else None
    guest_session_id = None
    if user_id is None:
        from app.guest_session_token import require_guest_session_id

        guest_session_id = require_guest_session_id(request)

    job = await get_job_for_owner(
        job_id, user_id=user_id, guest_session_id=guest_session_id
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Generation job not found.")
    return {
        "job_id": job.id,
        "status": job.status,
        "created_at": job.created_at,
        "final": job.final,
        "error": job.error,
        "events": job.events,
        "request_id": job.request_id,
    }


@router.get("/{job_id}/events")
async def generation_job_events(
    job_id: str,
    request: Request,
    token: Optional[str] = Query(default=None, description="JWT token for EventSource clients."),
):
    user_id, guest_session_id = await _resolve_sse_owner(request, token)
    live_job = JOBS.get(job_id)
    if live_job is not None and (
        (user_id is not None and live_job.user_id == user_id)
        or (guest_session_id and live_job.guest_session_id == guest_session_id)
    ):
        return StreamingResponse(
            stream_job_events(live_job),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    # Brief retry before 404: covers the tiny window between the create
    # commit on another worker and visibility here (or a tunnel/proxy replay).
    owned = None
    for attempt in range(4):
        owned = await get_job_for_owner(
            job_id, user_id=user_id, guest_session_id=guest_session_id
        )
        if owned is not None:
            break
        await asyncio.sleep(0.4 * (attempt + 1))
    if owned is None:
        raise HTTPException(status_code=404, detail="Generation job not found.")
    return StreamingResponse(
        stream_job_events_from_db(
            job_id, user_id=user_id, guest_session_id=guest_session_id
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/{job_id}/cancel")
async def cancel_generation_job(
    job_id: str,
    request: Request,
    current_user: User | None = Depends(get_optional_user),
):
    user_id = int(current_user.id) if current_user else None
    guest_session_id = None
    if user_id is None:
        from app.guest_session_token import require_guest_session_id

        guest_session_id = require_guest_session_id(request)

    if not await cancel_job(job_id, user_id=user_id, guest_session_id=guest_session_id):
        raise HTTPException(status_code=404, detail="Generation job not found.")
    return {"job_id": job_id, "status": "cancelling"}
