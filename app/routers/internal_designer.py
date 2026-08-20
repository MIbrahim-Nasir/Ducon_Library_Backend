"""
Internal Designer Agent tools API.

For separate internal tools (scripts, ops UIs) — not the public SPA.
Requires BOTH a valid designer-role user JWT and ``X-Internal-Tool-Key``.

POST   /internal/designer/jobs                 create and start a run
                                           (?wait=true or form wait=true to block)
GET    /internal/designer/jobs/{job_id}         inspect state (+ final_image_url when done)
GET    /internal/designer/jobs/{job_id}/events   SSE progress (Bearer or ?token=)
POST   /internal/designer/jobs/{job_id}/cancel   cancel

Job progress URLs also match the public designer API so tools may poll
``/designer/jobs/{id}`` with the same user JWT after create.
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.admin.settings_store import cfg, cfg_bool
from app.auth import require_designer, resolve_user_from_token
from app.config import IS_PRODUCTION
from app.db.database import async_session_maker
from app.db.models import User
from app.designer_agent import (
    JOBS,
    cancel_job,
    create_job,
    final_image_fields,
    get_job_for_user,
    run_designer_job,
    stream_job_events,
    stream_job_events_from_db,
)
from app.rate_limiter import require_rate_limit
from app.sse import SSE_HEADERS as _SSE_HEADERS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/designer", tags=["internal-designer"])

# Bot-hostile / human-ok defaults (overridable via settings / env).
_DEFAULT_RATE_MAX = 40
_DEFAULT_RATE_WINDOW_S = 3600  # 1 hour
_DEFAULT_MAX_CONCURRENT = 10
_DEFAULT_PROMPT_MAX_CHARS = 8000
_DEFAULT_WAIT_TIMEOUT_S = 15 * 60  # 15 minutes
_TERMINAL = frozenset({"completed", "failed", "cancelled"})

_IMAGE_MIME_PREFIX = "image/"
_MAGIC_SIGNATURES: dict[bytes, str] = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",
    b"\x00\x00\x00\x0cjP  ": "image/jp2",
    b"\x49\x49\x2a\x00": "image/tiff",
    b"\x4d\x4d\x00\x2a": "image/tiff",
}
_BLOCKED_EXTENSIONS = {".exe", ".sh", ".bat", ".cmd", ".js", ".py", ".php", ".rb", ".ps1"}


def _live_debug() -> bool:
    return cfg_bool("LIVE_DEBUG", False)


def _dbg(*args) -> None:
    if _live_debug():
        print(*args)


def _api_key_from_env() -> str:
    return os.getenv("INTERNAL_DESIGNER_API_KEY", "").strip()


def require_internal_designer_tool_key(
    x_internal_tool_key: str = Header("", alias="X-Internal-Tool-Key"),
) -> None:
    """
    Gate for internal tools: dedicated shared secret (in addition to designer JWT).

    Fail-closed when the env secret is unset (including production). Never log the key.
    """
    expected = _api_key_from_env()
    if not expected:
        raise HTTPException(
            status_code=403,
            detail="Internal designer API key not configured.",
        )
    provided = x_internal_tool_key or ""
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=403, detail="Forbidden.")


def _max_upload_bytes() -> int:
    return int(cfg("MAX_UPLOAD_SIZE_MB", 50)) * 1024 * 1024


def _prompt_max_chars() -> int:
    return int(cfg("INTERNAL_DESIGNER_PROMPT_MAX_CHARS", _DEFAULT_PROMPT_MAX_CHARS))


def _rate_max() -> int:
    return int(cfg("INTERNAL_DESIGNER_RATE_LIMIT", _DEFAULT_RATE_MAX))


def _rate_window_s() -> int:
    return int(cfg("INTERNAL_DESIGNER_RATE_WINDOW_SECONDS", _DEFAULT_RATE_WINDOW_S))


def _max_concurrent() -> int:
    return int(cfg("INTERNAL_DESIGNER_MAX_CONCURRENT", _DEFAULT_MAX_CONCURRENT))


def _wait_timeout_s() -> float:
    return float(cfg("INTERNAL_DESIGNER_WAIT_TIMEOUT_SECONDS", _DEFAULT_WAIT_TIMEOUT_S))


def _truthy_flag(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _want_wait(request: Request, form_wait: Optional[str]) -> bool:
    """Accept ``?wait=true`` query or multipart form ``wait=true``."""
    q = request.query_params.get("wait")
    if q is not None:
        return _truthy_flag(q)
    return _truthy_flag(form_wait)


def _job_status_payload(job) -> dict[str, Any]:
    """Shared GET/create-wait body: existing fields + top-level final image URL."""
    payload: dict[str, Any] = {
        "job_id": job.id,
        "status": job.status,
        "created_at": getattr(job, "created_at", None),
        "final": job.final,
        "error": job.error,
        "events": getattr(job, "events", None) or [],
    }
    if job.status == "completed":
        payload.update(final_image_fields(job.final))
    else:
        payload["final_image_url"] = None
        payload["final_image"] = None
    return payload


async def _await_job_terminal(job, *, timeout_s: float) -> None:
    """Poll in-memory job until terminal status or timeout (does not raise)."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    while job.status not in _TERMINAL:
        if time.monotonic() >= deadline:
            return
        task = job.task
        if task is not None and task.done():
            # Runner should have set terminal status; brief yield then re-check.
            await asyncio.sleep(0.05)
            if job.status in _TERMINAL:
                return
            # Task finished without terminal status — stop waiting.
            return
        await asyncio.sleep(0.5)


def _sniff_image_mime(data: bytes) -> str | None:
    header = data[:16]
    for sig, mime in _MAGIC_SIGNATURES.items():
        if header[: len(sig)] == sig:
            if sig == b"RIFF" and data[8:12] != b"WEBP":
                return None
            return mime
    return None


def _validate_image_upload(filename: str | None, data: bytes) -> None:
    """Raise 400/413 if the upload violates size or image MIME policy."""
    if not data:
        raise HTTPException(status_code=422, detail="user_image is empty.")
    max_up = _max_upload_bytes()
    if len(data) > max_up:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {max_up // (1024 * 1024)} MB.",
        )
    name = (filename or "").lower()
    ext = os.path.splitext(name)[1]
    if ext in _BLOCKED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not allowed.")
    sniffed = _sniff_image_mime(data)
    if sniffed is None or not sniffed.startswith(_IMAGE_MIME_PREFIX):
        raise HTTPException(status_code=400, detail="File type not allowed.")


def _validate_prompt(prompt: Optional[str]) -> Optional[str]:
    if prompt is None:
        return None
    text = prompt.strip()
    if not text:
        return None
    max_chars = _prompt_max_chars()
    if len(text) > max_chars:
        raise HTTPException(
            status_code=422,
            detail=f"prompt too long (max {max_chars} characters).",
        )
    return text


def _count_active_jobs_for_user(user_id: int) -> int:
    terminal = {"completed", "failed", "cancelled"}
    return sum(
        1
        for job in JOBS.values()
        if job.user_id == user_id and job.status not in terminal
    )


async def _resolve_sse_user(request: Request, token: Optional[str]) -> User:
    raw_token = token
    if not raw_token:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            raw_token = auth.split(" ", 1)[1].strip()

    if not raw_token:
        raise HTTPException(status_code=401, detail="Authentication required.")

    async with async_session_maker() as db:
        user = await resolve_user_from_token(raw_token, db)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    from app.auth import is_designer_role

    if not is_designer_role(user.role):
        raise HTTPException(status_code=403, detail="Designer role required")
    return user


@router.post("/jobs")
async def create_internal_designer_job(
    request: Request,
    prompt: Optional[str] = Form(
        None,
        description="User design goal/suggestions. If omitted, agent designs independently.",
    ),
    model: str = Form("flash", description='"flash" or "pro" image generation model.'),
    aspect_ratio: Optional[str] = Form("16:9", description='Output aspect ratio, e.g. "16:9".'),
    wait: Optional[str] = Form(
        None,
        description='If "true", block until job completes/fails/cancels or wait timeout.',
    ),
    user_image: UploadFile = File(..., description="Client space image to redesign."),
    current_user: User = Depends(require_designer),
    _: None = Depends(require_internal_designer_tool_key),
):
    """
    Start a designer agent job from an internal tool.

    Headers required:
      Authorization: Bearer <user JWT>   (user.role must be designer)
      X-Internal-Tool-Key: <INTERNAL_DESIGNER_API_KEY>

    Optional wait mode (``?wait=true`` or form ``wait=true``): holds the HTTP
    response until the job reaches a terminal status or
    ``INTERNAL_DESIGNER_WAIT_TIMEOUT_SECONDS`` (default 900). On timeout the
    response still includes ``job_id`` with ``status`` typically ``running``
    so the client can poll. When completed, includes ``final_image_url``.
    """
    await require_rate_limit(
        request,
        max_requests=_rate_max(),
        window_seconds=_rate_window_s(),
        key_prefix="internal_designer",
        key_suffix=f"u:{current_user.id}",
    )

    active = _count_active_jobs_for_user(int(current_user.id))
    if active >= _max_concurrent():
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent designer jobs (max {_max_concurrent()}).",
        )

    raw = await user_image.read()
    _validate_image_upload(user_image.filename, raw)
    prompt = _validate_prompt(prompt)

    if model not in {"flash", "pro"}:
        raise HTTPException(status_code=422, detail="model must be 'flash' or 'pro'.")

    embedding_model = request.app.state.embedding_model
    collection = request.app.state.collection
    do_wait = _want_wait(request, wait)

    job = create_job(current_user)
    _dbg(
        "[INTERNAL DESIGNER ▶ CREATE]",
        {
            "job_id": job.id,
            "user_id": current_user.id,
            "prompt_chars": len(prompt or ""),
            "model": model,
            "aspect_ratio": aspect_ratio,
            "upload_bytes": len(raw),
            "wait": do_wait,
            "is_production": IS_PRODUCTION,
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

    job.task = asyncio.create_task(_run(), name=f"internal-designer-job-{job.id[:8]}")

    # Public + internal status URLs (same job id / ownership).
    base = {
        "job_id": job.id,
        "status": job.status,
        "events_url": f"/designer/jobs/{job.id}/events",
        "status_url": f"/designer/jobs/{job.id}",
        "internal_events_url": f"/internal/designer/jobs/{job.id}/events",
        "internal_status_url": f"/internal/designer/jobs/{job.id}",
        "final_image_url": None,
        "final_image": None,
    }

    if not do_wait:
        return base

    await _await_job_terminal(job, timeout_s=_wait_timeout_s())
    base["status"] = job.status
    base["final"] = job.final
    base["error"] = job.error
    if job.status == "completed":
        base.update(final_image_fields(job.final))
    elif job.status in _TERMINAL:
        base["final_image_url"] = None
        base["final_image"] = None
    # else still running/queued after timeout — client should poll
    return base


@router.get("/jobs/{job_id}")
async def get_internal_designer_job(
    job_id: str,
    current_user: User = Depends(require_designer),
    _: None = Depends(require_internal_designer_tool_key),
):
    job = await get_job_for_user(job_id, int(current_user.id))
    if job is None:
        raise HTTPException(status_code=404, detail="Designer job not found.")
    return _job_status_payload(job)


@router.get("/jobs/{job_id}/events")
async def internal_designer_job_events(
    job_id: str,
    request: Request,
    token: Optional[str] = Query(default=None, description="JWT token for EventSource clients."),
    _: None = Depends(require_internal_designer_tool_key),
):
    current_user = await _resolve_sse_user(request, token)
    uid = int(current_user.id)
    live_job = JOBS.get(job_id)
    if live_job is not None and live_job.user_id == uid:
        return StreamingResponse(
            stream_job_events(live_job),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    owned = await get_job_for_user(job_id, uid)
    if owned is None:
        raise HTTPException(status_code=404, detail="Designer job not found.")
    return StreamingResponse(
        stream_job_events_from_db(job_id, uid),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_internal_designer_job(
    job_id: str,
    current_user: User = Depends(require_designer),
    _: None = Depends(require_internal_designer_tool_key),
):
    if not await cancel_job(job_id, int(current_user.id)):
        raise HTTPException(status_code=404, detail="Designer job not found.")
    return {"job_id": job_id, "status": "cancelling"}
