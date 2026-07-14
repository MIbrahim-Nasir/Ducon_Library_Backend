"""
Multi-image generation jobs — create / poll / SSE / cancel.

Mirrors designer_jobs: the run lives on one gunicorn worker with an in-memory
queue; Postgres ``generation_jobs`` lets any worker serve GET + poll-based SSE.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, text

from app.db.database import async_session_maker
from app.db.models import GenerationJobRow, GuestSession
from app.error_logger import log_error
from app.tool_generate_image import ImageDescriptor, generate_multi_image

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@dataclass
class GenerationJob:
    id: str
    user_id: Optional[int] = None
    guest_session_id: Optional[str] = None
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    events: list[dict[str, Any]] = field(default_factory=list)
    queue: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)
    task: Optional[asyncio.Task] = None
    final: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    request_id: Optional[str] = None
    cancel_requested: bool = False


JOBS: dict[str, GenerationJob] = {}


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _owns(job: GenerationJob, *, user_id: Optional[int], guest_session_id: Optional[str]) -> bool:
    if user_id is not None and job.user_id == int(user_id):
        return True
    if guest_session_id and job.guest_session_id == guest_session_id:
        return True
    return False


async def persist_job_create(job: GenerationJob) -> bool:
    """Insert the job row. Called from the create route BEFORE the response is
    returned so a cross-worker /events request can always find the row.
    Returns False when the insert failed (caller should abort the job)."""
    try:
        async with async_session_maker() as db:
            db.add(
                GenerationJobRow(
                    id=job.id,
                    user_id=job.user_id,
                    guest_session_id=job.guest_session_id,
                    status=job.status,
                    request_id=job.request_id,
                )
            )
            await db.commit()
        return True
    except Exception:
        logger.exception("[GenerationJob] persist create failed job_id=%s", job.id)
        return False


async def _persist_job_event(job: GenerationJob, payload: dict[str, Any]) -> None:
    try:
        async with async_session_maker() as db:
            await db.execute(
                text(
                    "UPDATE generation_jobs "
                    "SET events = events || CAST(:evt AS jsonb), "
                    "    status = :status, "
                    "    updated_at = NOW() "
                    "WHERE id = :id"
                ),
                {
                    "evt": json.dumps([payload], ensure_ascii=False),
                    "status": job.status,
                    "id": job.id,
                },
            )
            await db.commit()
    except Exception:
        logger.exception("[GenerationJob] persist event failed job_id=%s", job.id)


async def _persist_job_final(
    job: GenerationJob,
    *,
    final: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    try:
        async with async_session_maker() as db:
            row = await db.get(GenerationJobRow, job.id)
            if row is None:
                # Create failed earlier — still write a terminal row so
                # cross-worker pollers do not wait forever.
                row = GenerationJobRow(
                    id=job.id,
                    user_id=job.user_id,
                    guest_session_id=job.guest_session_id,
                    status=job.status,
                    request_id=job.request_id,
                    events=list(job.events or []),
                )
                db.add(row)
            row.status = job.status
            row.updated_at = datetime.now(timezone.utc)
            if final is not None:
                row.final = final
            if error is not None:
                row.error = error
            await db.commit()
    except Exception:
        logger.exception("[GenerationJob] persist final failed job_id=%s", job.id)


async def emit(job: GenerationJob, event_type: str, **data: Any) -> None:
    payload = {
        "type": event_type,
        "job_id": job.id,
        "status": job.status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    job.events.append(payload)
    await job.queue.put(_sse(payload))
    await _persist_job_event(job, payload)


def create_job(
    *,
    user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> GenerationJob:
    if user_id is None and not guest_session_id:
        raise ValueError("user_id or guest_session_id required")
    job = GenerationJob(
        id=uuid.uuid4().hex,
        user_id=int(user_id) if user_id is not None else None,
        guest_session_id=guest_session_id,
        request_id=request_id,
    )
    JOBS[job.id] = job
    return job


def _row_to_job(row: GenerationJobRow) -> GenerationJob:
    restored = GenerationJob(
        id=row.id,
        user_id=int(row.user_id) if row.user_id is not None else None,
        guest_session_id=row.guest_session_id,
        request_id=row.request_id,
    )
    restored.status = row.status
    restored.created_at = row.created_at.isoformat() if row.created_at else restored.created_at
    restored.events = list(row.events or [])
    restored.final = row.final
    restored.error = row.error
    restored.cancel_requested = bool(row.cancel_requested)
    restored.task = None
    return restored


async def get_job_for_owner(
    job_id: str,
    *,
    user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None,
) -> Optional[GenerationJob]:
    job = JOBS.get(job_id)
    if job is not None and _owns(job, user_id=user_id, guest_session_id=guest_session_id):
        return job
    try:
        async with async_session_maker() as db:
            row = await db.get(GenerationJobRow, job_id)
            if row is None:
                return None
            restored = _row_to_job(row)
            if not _owns(restored, user_id=user_id, guest_session_id=guest_session_id):
                return None
            return restored
    except Exception:
        logger.exception("[GenerationJob] get_job_for_owner failed job_id=%s", job_id)
        return None


async def stream_job_events(job: GenerationJob):
    yield ": connected\n\n"
    for event in job.events:
        yield _sse(event)
    while True:
        try:
            item = await asyncio.wait_for(job.queue.get(), timeout=10)
        except asyncio.TimeoutError:
            yield ": keep-alive\n\n"
            continue
        if item is None:
            break
        yield item


async def stream_job_events_from_db(
    job_id: str,
    *,
    user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None,
):
    yield ": connected\n\n"
    sent = 0
    missing_reads = 0
    # Hard cap: a job stranded non-terminal (worker crash before the stale
    # sweeper runs) must not keep this poller alive forever.
    deadline = asyncio.get_event_loop().time() + 30 * 60
    while asyncio.get_event_loop().time() < deadline:
        try:
            async with async_session_maker() as db:
                row = await db.get(GenerationJobRow, job_id)
        except Exception:
            yield ": keep-alive\n\n"
            await asyncio.sleep(2)
            continue
        if row is None:
            # Tolerate a brief window where the create-commit hasn't landed yet
            # (or a replica lag) instead of silently ending the stream.
            missing_reads += 1
            if missing_reads > 5:
                yield _sse({"type": "error", "job_id": job_id,
                            "message": "Generation job not found."})
                break
            yield ": keep-alive\n\n"
            await asyncio.sleep(1)
            continue
        missing_reads = 0
        restored = _row_to_job(row)
        if not _owns(restored, user_id=user_id, guest_session_id=guest_session_id):
            break
        events = list(row.events or [])
        if len(events) > sent:
            for event in events[sent:]:
                yield _sse(event)
            sent = len(events)
        if row.status in _TERMINAL_STATUSES:
            break
        await asyncio.sleep(2)


async def cancel_job(
    job_id: str,
    *,
    user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None,
) -> bool:
    job = JOBS.get(job_id)
    if job is not None and _owns(job, user_id=user_id, guest_session_id=guest_session_id):
        job.cancel_requested = True
        if job.task and not job.task.done():
            job.task.cancel()
        await _persist_cancel_request(job_id, user_id=user_id, guest_session_id=guest_session_id)
        return True

    owned = await get_job_for_owner(
        job_id, user_id=user_id, guest_session_id=guest_session_id
    )
    if owned is None:
        return False
    await _persist_cancel_request(job_id, user_id=user_id, guest_session_id=guest_session_id)
    return True


async def _persist_cancel_request(
    job_id: str,
    *,
    user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None,
) -> None:
    try:
        async with async_session_maker() as db:
            if user_id is not None:
                await db.execute(
                    text(
                        "UPDATE generation_jobs "
                        "SET cancel_requested = TRUE, updated_at = NOW() "
                        "WHERE id = :id AND user_id = :uid"
                    ),
                    {"id": job_id, "uid": user_id},
                )
            elif guest_session_id:
                await db.execute(
                    text(
                        "UPDATE generation_jobs "
                        "SET cancel_requested = TRUE, updated_at = NOW() "
                        "WHERE id = :id AND guest_session_id = :gid"
                    ),
                    {"id": job_id, "gid": guest_session_id},
                )
            await db.commit()
    except Exception:
        logger.exception("[GenerationJob] persist cancel failed job_id=%s", job_id)


async def run_generation_job(
    *,
    job: GenerationJob,
    prompt: str,
    descriptors: list[ImageDescriptor],
    model: str,
    aspect_ratio: Optional[str],
) -> None:
    """Background worker: generate_multi_image + progress events.

    The job row is already persisted by the create route (persist_job_create)
    before this task starts, so cross-worker /events readers never race it.
    """
    try:
        job.status = "running"
        await emit(job, "status", message="Generation started")

        if job.cancel_requested:
            job.status = "cancelled"
            await emit(job, "cancelled", message="Cancelled before start")
            await _persist_job_final(job, error="cancelled")
            await job.queue.put(None)
            return

        async with async_session_maker() as db:
            guest_session = None
            if job.guest_session_id:
                result = await db.execute(
                    select(GuestSession).where(GuestSession.session_id == job.guest_session_id)
                )
                guest_session = result.scalar_one_or_none()
                if guest_session is None:
                    raise RuntimeError("Guest session was not found before saving generation.")

            result = await generate_multi_image(
                user_id=job.user_id,
                guest_session_id=job.guest_session_id,
                guest_session=guest_session,
                prompt=prompt,
                descriptors=descriptors,
                model=model,
                aspect_ratio=aspect_ratio or None,
                db=db,
                enable_verify=True,
            )
            await db.commit()

        if job.cancel_requested:
            job.status = "cancelled"
            await emit(job, "cancelled", message="Cancelled after generation")
            await _persist_job_final(job, error="cancelled")
        else:
            job.status = "completed"
            job.final = result
            await emit(job, "done", **result)
            await _persist_job_final(job, final=result)
    except asyncio.CancelledError:
        job.status = "cancelled"
        job.error = "cancelled"
        await emit(job, "cancelled", message="Cancelled")
        await _persist_job_final(job, error="cancelled")
        raise
    except (ValueError, RuntimeError) as exc:
        job.status = "failed"
        job.error = str(exc)
        await log_error(
            "multi_image",
            "generation_jobs.worker",
            str(exc),
            user_id=job.user_id,
            guest_session_id=job.guest_session_id,
            endpoint="/generation/jobs",
            request_id=job.request_id,
            exc=exc,
        )
        await emit(job, "error", message=str(exc))
        await _persist_job_final(job, error=str(exc))
    except Exception as exc:
        logger.exception("[GenerationJob] unexpected error job_id=%s", job.id)
        job.status = "failed"
        job.error = f"Unexpected error: {exc}"
        await log_error(
            "multi_image",
            "generation_jobs.worker",
            "Unexpected generation job error",
            user_id=job.user_id,
            guest_session_id=job.guest_session_id,
            endpoint="/generation/jobs",
            request_id=job.request_id,
            exc=exc,
        )
        await emit(job, "error", message=job.error)
        await _persist_job_final(job, error=job.error)
    finally:
        try:
            await job.queue.put(None)
        except Exception:
            pass
        # Keep completed jobs briefly for same-worker SSE; cross-worker uses DB.
        if job.status in _TERMINAL_STATUSES:
            await asyncio.sleep(0)
            JOBS.pop(job.id, None)
