"""Designer job store, SSE emit, persist, cancel, and drain."""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import text

from app import storage
from app.admin.settings_store import cfg
from app.db.database import async_session_maker
from app.db.models import DesignerJobRow, User

_LIVE_DEBUG: bool = False
_SHUTDOWN_INTERRUPT_MSG = "Server restarted (reload); job interrupted."
_USER_CANCEL_MSG = "Designer job was cancelled."
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _dbg(*args) -> None:
    if cfg("LIVE_DEBUG", _LIVE_DEBUG):
        print(*args)


@dataclass
class DesignerJob:
    id: str
    user_id: int
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    events: list[dict[str, Any]] = field(default_factory=list)
    queue: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)
    task: Optional[asyncio.Task] = None
    final: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    cancel_requested: bool = False


JOBS: dict[str, DesignerJob] = {}


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def enrich_generation_record(gen: dict[str, Any]) -> dict[str, Any]:
    """Return generation dict with a fresh presigned URL for the UI."""
    enriched = dict(gen)
    gen_id = enriched.get("id")
    if gen_id is not None:
        enriched["signed_url"] = storage.get_generation_url(int(gen_id), enriched.get("url"))
    return enriched


_enrich_generation_record = enrich_generation_record


def final_image_fields(final: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Top-level final image fields for API clients (Revit / internal tools)."""
    empty: dict[str, Any] = {"final_image_url": None, "final_image": None}
    if not isinstance(final, dict):
        return empty
    best = final.get("best_generation")
    if not isinstance(best, dict) or not best:
        return empty
    enriched = enrich_generation_record(best)
    url = enriched.get("signed_url") or enriched.get("url")
    return {
        "final_image_url": url,
        "final_image": {
            "id": enriched.get("id"),
            "url": enriched.get("url"),
            "signed_url": enriched.get("signed_url"),
            "generation_name": enriched.get("generation_name"),
        },
    }


async def _persist_job_create(job: DesignerJob) -> None:
    try:
        async with async_session_maker() as db:
            db.add(DesignerJobRow(
                id=job.id,
                user_id=job.user_id,
                status=job.status,
            ))
            await db.commit()
    except Exception:
        _dbg("[DESIGNER ⚠ persist create failed]")


async def _persist_job_event(job: DesignerJob, payload: dict[str, Any]) -> None:
    try:
        async with async_session_maker() as db:
            await db.execute(
                text(
                    "UPDATE designer_jobs "
                    "SET events = events || CAST(:evt AS jsonb), "
                    "    status = :status, "
                    "    updated_at = NOW() "
                    "WHERE id = :id"
                ),
                {"evt": json.dumps([payload], ensure_ascii=False), "status": job.status, "id": job.id},
            )
            await db.commit()
    except Exception:
        _dbg("[DESIGNER ⚠ persist event failed]")


async def _persist_job_final(
    job: DesignerJob,
    *,
    final: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    try:
        async with async_session_maker() as db:
            row = await db.get(DesignerJobRow, job.id)
            if row is None:
                row = DesignerJobRow(
                    id=job.id,
                    user_id=job.user_id,
                    status=job.status,
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
        _dbg("[DESIGNER ⚠ persist final failed]")


async def emit(job: DesignerJob, event_type: str, **data: Any) -> None:
    payload = {
        "type": event_type,
        "job_id": job.id,
        "status": job.status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    _dbg("[DESIGNER EVENT]", payload)
    job.events.append(payload)
    await job.queue.put(_sse(payload))
    await _persist_job_event(job, payload)


def create_job(user: User) -> DesignerJob:
    job = DesignerJob(id=uuid.uuid4().hex, user_id=int(user.id))
    JOBS[job.id] = job
    return job


async def get_job_for_user(job_id: str, user_id: int) -> Optional[DesignerJob]:
    job = JOBS.get(job_id)
    if job is not None and job.user_id == user_id:
        return job
    try:
        async with async_session_maker() as db:
            row = await db.get(DesignerJobRow, job_id)
            if row is None or row.user_id != user_id:
                return None
            restored = DesignerJob(id=row.id, user_id=int(row.user_id))
            restored.status = row.status
            restored.created_at = row.created_at.isoformat() if row.created_at else restored.created_at
            restored.events = list(row.events or [])
            restored.final = row.final
            restored.error = row.error
            restored.cancel_requested = bool(row.cancel_requested)
            restored.task = None
            return restored
    except Exception:
        _dbg("[DESIGNER ⚠ get_job_for_user db read failed]")
        return None


async def drain_jobs_on_shutdown(*, timeout: float = 10.0) -> int:
    """Cancel in-flight designer jobs so CancelledError handlers persist terminal status."""
    in_flight = [
        j.task
        for j in list(JOBS.values())
        if j.task is not None and not j.task.done() and j.status not in _TERMINAL_STATUSES
    ]
    for job in list(JOBS.values()):
        if job.task is not None and not job.task.done() and not job.cancel_requested:
            job.error = _SHUTDOWN_INTERRUPT_MSG
    for t in in_flight:
        t.cancel()
    if in_flight:
        await asyncio.wait(in_flight, timeout=timeout)
    return len(in_flight)


async def stream_job_events(job: DesignerJob):
    """Live SSE stream for a job owned by THIS worker."""
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


async def stream_job_events_from_db(job_id: str, user_id: int):
    """Polling SSE stream for a job owned by ANOTHER worker."""
    yield ": connected\n\n"
    sent = 0
    while True:
        try:
            async with async_session_maker() as db:
                row = await db.get(DesignerJobRow, job_id)
        except Exception:
            yield ": keep-alive\n\n"
            await asyncio.sleep(2)
            continue
        if row is None or row.user_id != user_id:
            break
        events = list(row.events or [])
        if len(events) > sent:
            for event in events[sent:]:
                yield _sse(event)
            sent = len(events)
        if row.status in _TERMINAL_STATUSES:
            break
        await asyncio.sleep(2)


async def cancel_job(job_id: str, user_id: int) -> bool:
    job = JOBS.get(job_id)
    if job is not None and job.user_id == user_id:
        job.cancel_requested = True
        if job.task and not job.task.done():
            job.task.cancel()
        await _persist_cancel_request(job_id, user_id)
        return True

    try:
        async with async_session_maker() as db:
            row = await db.get(DesignerJobRow, job_id)
            if row is None or row.user_id != user_id:
                return False
        await _persist_cancel_request(job_id, user_id)
        return True
    except Exception:
        _dbg("[DESIGNER ⚠ cross-worker cancel db read failed]")
        return False


async def _persist_cancel_request(job_id: str, user_id: int) -> None:
    try:
        async with async_session_maker() as db:
            await db.execute(
                text(
                    "UPDATE designer_jobs "
                    "SET cancel_requested = TRUE, updated_at = NOW() "
                    "WHERE id = :id AND user_id = :uid"
                ),
                {"id": job_id, "uid": user_id},
            )
            await db.commit()
    except Exception:
        _dbg("[DESIGNER ⚠ persist cancel failed]")


def check_cancelled(job: DesignerJob) -> None:
    if job.cancel_requested:
        raise asyncio.CancelledError()


_check_cancelled = check_cancelled
