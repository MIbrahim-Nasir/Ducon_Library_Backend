"""Non-blocking structured application error logging to Postgres.

Call sites use ``log_error`` / ``log_warning``; entries are enqueued and
batch-inserted by a background writer (same pattern as UsageRecorder).
Logging failures never propagate to the caller.
"""
from __future__ import annotations

import asyncio
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import async_session_maker
from app.db.models import AppErrorLog

logger = logging.getLogger(__name__)

ErrorCategory = Literal[
    "generation",
    "chat",
    "voice",
    "studio",
    "multi_image",
    "gemini",
    "storage",
    "auth",
    "guest",
    "admin",
    "other",
]
ErrorSeverity = Literal["error", "warning"]

_QUEUE_MAXSIZE = 5_000
_FLUSH_INTERVAL_S = 2.0
_FLUSH_BATCH_SIZE = 100


@dataclass
class ErrorLogEntry:
    severity: ErrorSeverity
    category: ErrorCategory
    source: str
    message: str
    user_id: Optional[int] = None
    guest_session_id: Optional[str] = None
    endpoint: Optional[str] = None
    model: Optional[str] = None
    error_type: Optional[str] = None
    detail: Optional[dict[str, Any]] = None
    request_id: Optional[str] = None
    http_status: Optional[int] = None
    provider_status: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _exc_detail(exc: BaseException | None, extra: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    out: dict[str, Any] = dict(extra or {})
    if exc is not None:
        out.setdefault("exception_type", type(exc).__name__)
        out.setdefault("exception_message", str(exc))
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        if tb:
            out["traceback"] = "".join(tb)[-8000:]
    return out or None


class ErrorLogger:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[ErrorLogEntry] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._writer_task: asyncio.Task | None = None
        self._running = False
        self._written = 0
        self._dropped = 0

    def stats(self) -> dict[str, int]:
        return {
            "queue_depth": self._queue.qsize(),
            "written": self._written,
            "dropped": self._dropped,
        }

    def record(self, entry: ErrorLogEntry) -> None:
        if not self._running:
            return
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            self._dropped += 1

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._writer_task = asyncio.create_task(self._writer_loop(), name="error-log-writer")

    async def stop(self) -> None:
        self._running = False
        await self._flush()
        if self._writer_task:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except (asyncio.CancelledError, Exception):
                pass
            self._writer_task = None

    async def _writer_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(_FLUSH_INTERVAL_S)
                await self._flush()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("error log writer loop error")
                await asyncio.sleep(_FLUSH_INTERVAL_S)

    async def _flush(self) -> None:
        if self._queue.empty():
            return
        entries: list[ErrorLogEntry] = []
        while not self._queue.empty() and len(entries) < _FLUSH_BATCH_SIZE * 4:
            try:
                entries.append(self._queue.get_nowait())
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        if not entries:
            return
        try:
            async with async_session_maker() as db:
                await self._insert_entries(db, entries)
                await db.commit()
            self._written += len(entries)
        except Exception:
            logger.exception("failed to batch-insert %d error log entries", len(entries))

    async def _insert_entries(self, db: AsyncSession, entries: list[ErrorLogEntry]) -> None:
        rows = [
            AppErrorLog(
                severity=e.severity,
                category=e.category,
                source=e.source,
                message=e.message[:4000],
                user_id=e.user_id,
                guest_session_id=e.guest_session_id,
                endpoint=e.endpoint,
                model=e.model,
                error_type=e.error_type,
                detail=e.detail,
                request_id=e.request_id,
                http_status=e.http_status,
                provider_status=e.provider_status,
                created_at=e.created_at,
            )
            for e in entries
        ]
        db.add_all(rows)


_logger = ErrorLogger()


def get_error_logger() -> ErrorLogger:
    return _logger


def _resolve_request_id(request_id: Optional[str]) -> Optional[str]:
    """Prefer an explicit id; otherwise take the middleware ContextVar."""
    if request_id:
        return request_id
    try:
        from app.middleware.request_id import get_request_id

        return get_request_id()
    except Exception:
        return None


async def log_error(
    category: ErrorCategory,
    source: str,
    message: str,
    *,
    user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    exc: BaseException | None = None,
    extra: Optional[dict[str, Any]] = None,
    request_id: Optional[str] = None,
    http_status: Optional[int] = None,
    provider_status: Optional[int] = None,
    error_type: Optional[str] = None,
) -> None:
    """Enqueue an error log entry. Never raises."""
    try:
        _logger.record(
            ErrorLogEntry(
                severity="error",
                category=category,
                source=source,
                message=message,
                user_id=user_id,
                guest_session_id=guest_session_id,
                endpoint=endpoint,
                model=model,
                error_type=error_type or (type(exc).__name__ if exc else None),
                detail=_exc_detail(exc, extra),
                request_id=_resolve_request_id(request_id),
                http_status=http_status,
                provider_status=provider_status,
            )
        )
    except Exception:
        logger.exception("log_error enqueue failed")


async def log_warning(
    category: ErrorCategory,
    source: str,
    message: str,
    *,
    user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    exc: BaseException | None = None,
    extra: Optional[dict[str, Any]] = None,
    request_id: Optional[str] = None,
    http_status: Optional[int] = None,
    provider_status: Optional[int] = None,
    error_type: Optional[str] = None,
) -> None:
    """Enqueue a warning-level log entry. Never raises."""
    try:
        _logger.record(
            ErrorLogEntry(
                severity="warning",
                category=category,
                source=source,
                message=message,
                user_id=user_id,
                guest_session_id=guest_session_id,
                endpoint=endpoint,
                model=model,
                error_type=error_type or (type(exc).__name__ if exc else None),
                detail=_exc_detail(exc, extra),
                request_id=_resolve_request_id(request_id),
                http_status=http_status,
                provider_status=provider_status,
            )
        )
    except Exception:
        logger.exception("log_warning enqueue failed")
