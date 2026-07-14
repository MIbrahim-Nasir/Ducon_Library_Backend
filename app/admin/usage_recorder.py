"""Non-blocking AI usage / cost tracking.

Agent call sites call ``record(...)`` with token counts + model + agent name.
The event is enqueued onto an asyncio.Queue (put_nowait — never blocks). A
single background writer task drains the queue every 2 seconds OR when 200
events accumulate, and batch-inserts into api_usage_events.

If the queue is full (e.g. DB is slow), events are dropped + a dropped counter
is incremented so the admin dashboard can surface "dropped events" rather than
silently losing data. The request path is never blocked.

A separate hourly task recomputes usage_daily_rollup for fast dashboard queries.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta, timezone
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.pricing import compute_cost
from app.db.database import async_session_maker
from app.db.models import ApiUsageEvent, SessionActivity, UsageDailyRollup

logger = logging.getLogger(__name__)

_QUEUE_MAXSIZE = 10_000
_FLUSH_INTERVAL_S = 2.0
_FLUSH_BATCH_SIZE = 200
_ROLLUP_INTERVAL_S = 3600.0  # hourly


@dataclass
class UsageEvent:
    agent: str
    model: str
    provider: str = "gemini"
    user_id: Optional[int] = None
    guest_session_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    image_count: int = 0
    audio_input_tokens: int = 0
    latency_ms: Optional[int] = None
    status: str = "success"
    error_message: Optional[str] = None
    # cost computed at record() time from pricing table
    cost_usd: float = 0.0
    # for session heartbeats only
    is_heartbeat: bool = False
    occurred_at: float = field(default_factory=time.time)


class UsageRecorder:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[UsageEvent] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._writer_task: Optional[asyncio.Task] = None
        self._rollup_task: Optional[asyncio.Task] = None
        self._dropped = 0
        self._written = 0
        self._running = False

    # ── public API ─────────────────────────────────────────────────────────

    def record(
        self,
        *,
        agent: str,
        model: str,
        provider: str = "gemini",
        user_id: Optional[int] = None,
        guest_session_id: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        image_count: int = 0,
        audio_input_tokens: int = 0,
        latency_ms: Optional[int] = None,
        status: str = "success",
        error_message: Optional[str] = None,
    ) -> None:
        """Enqueue a usage event. Non-blocking; never raises into the caller."""
        if not self._running:
            return
        cost = compute_cost(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            image_count=image_count,
            audio_input_tokens=audio_input_tokens,
        )
        ev = UsageEvent(
            agent=agent,
            model=model,
            provider=provider,
            user_id=user_id,
            guest_session_id=guest_session_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            image_count=image_count,
            audio_input_tokens=audio_input_tokens,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
            cost_usd=cost,
        )
        try:
            self._queue.put_nowait(ev)
        except asyncio.QueueFull:
            self._dropped += 1
            logger.warning("usage queue full — dropped event (total dropped: %d)", self._dropped)

    def heartbeat(self, *, user_id: Optional[int] = None, guest_session_id: Optional[str] = None) -> None:
        if not self._running:
            return
        ev = UsageEvent(
            agent="session",
            model="internal",
            provider="internal",
            user_id=user_id,
            guest_session_id=guest_session_id,
            is_heartbeat=True,
        )
        try:
            self._queue.put_nowait(ev)
        except asyncio.QueueFull:
            self._dropped += 1

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._writer_task = asyncio.create_task(self._writer_loop(), name="usage-writer")
        self._rollup_task = asyncio.create_task(self._rollup_loop(), name="usage-rollup")

    async def stop(self) -> None:
        self._running = False
        # final flush
        await self._flush()
        for t in (self._writer_task, self._rollup_task):
            if t:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        self._writer_task = None
        self._rollup_task = None

    # ── internal ───────────────────────────────────────────────────────────

    async def _writer_loop(self) -> None:
        while self._running:
            try:
                # Always sleep for the flush interval. We previously used
                # ``asyncio.wait_for(self._queue.join(), timeout=...)`` but on an
                # empty queue ``Queue.join()`` returns immediately, and the
                # ``await`` never suspends — producing a tight busy-spin that
                # starves the event loop and prevents uvicorn from completing
                # startup. ``asyncio.sleep`` always yields, so other tasks
                # (including uvicorn's own startup) can run.
                await asyncio.sleep(_FLUSH_INTERVAL_S)
                await self._flush()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("usage writer loop error")
                await asyncio.sleep(_FLUSH_INTERVAL_S)

    async def _flush(self) -> None:
        if self._queue.empty():
            return
        events: list[UsageEvent] = []
        while not self._queue.empty() and len(events) < _FLUSH_BATCH_SIZE * 4:
            try:
                events.append(self._queue.get_nowait())
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        if not events:
            return
        try:
            async with async_session_maker() as db:
                await self._insert_events(db, events)
                await db.commit()
            self._written += len(events)
        except Exception:
            logger.exception("failed to batch-insert %d usage events", len(events))
            from app.error_logger import log_error
            await log_error(
                "admin",
                "usage_recorder._flush",
                f"Failed to batch-insert {len(events)} usage events",
            )
            # don't re-enqueue (could loop forever); they're lost but logged

    async def _insert_events(self, db: AsyncSession, events: list[UsageEvent]) -> None:
        from datetime import datetime
        usage_rows: list[ApiUsageEvent] = []
        heartbeat_rows: list[SessionActivity] = []
        now = datetime.now(timezone.utc)
        for ev in events:
            if ev.is_heartbeat:
                heartbeat_rows.append(SessionActivity(
                    user_id=ev.user_id,
                    guest_session_id=ev.guest_session_id,
                    occurred_at=now,
                ))
                continue
            usage_rows.append(ApiUsageEvent(
                user_id=ev.user_id,
                guest_session_id=ev.guest_session_id,
                agent=ev.agent,
                model=ev.model,
                provider=ev.provider,
                input_tokens=ev.input_tokens,
                output_tokens=ev.output_tokens,
                image_count=ev.image_count,
                cost_usd=ev.cost_usd,
                latency_ms=ev.latency_ms,
                status=ev.status,
                error_message=ev.error_message,
            ))
        if usage_rows:
            db.add_all(usage_rows)
        if heartbeat_rows:
            db.add_all(heartbeat_rows)

    async def _rollup_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(_ROLLUP_INTERVAL_S)
                await self._recompute_rollup()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("usage rollup error")

    async def _recompute_rollup(self) -> None:
        """Aggregate api_usage_events for the last 2 days into usage_daily_rollup.

        Uses an upsert pattern so re-runs are idempotent. We recompute a rolling
        2-day window so late-arriving events (writer lag) get captured.
        """
        try:
            async with async_session_maker() as db:
                # Idempotent upsert via raw SQL (postgres ON CONFLICT)
                sql = text("""
                    INSERT INTO usage_daily_rollup
                        (day, user_id, agent, model,
                         total_calls, total_input_tokens, total_output_tokens,
                         total_image_count, total_cost_usd)
                    SELECT
                        DATE(created_at) AS day,
                        user_id,
                        agent,
                        model,
                        COUNT(*) AS total_calls,
                        COALESCE(SUM(input_tokens), 0) AS total_input_tokens,
                        COALESCE(SUM(output_tokens), 0) AS total_output_tokens,
                        COALESCE(SUM(image_count), 0) AS total_image_count,
                        COALESCE(SUM(cost_usd), 0) AS total_cost_usd
                    FROM api_usage_events
                    WHERE created_at >= NOW() - INTERVAL '2 days'
                    GROUP BY DATE(created_at), user_id, agent, model
                    ON CONFLICT (day, user_id, agent, model)
                    DO UPDATE SET
                        total_calls = EXCLUDED.total_calls,
                        total_input_tokens = EXCLUDED.total_input_tokens,
                        total_output_tokens = EXCLUDED.total_output_tokens,
                        total_image_count = EXCLUDED.total_image_count,
                        total_cost_usd = EXCLUDED.total_cost_usd
                """)
                await db.execute(sql)
                await db.commit()
        except Exception:
            logger.exception("rollup recompute failed")

    # ── introspection ──────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "running": self._running,
            "queue_depth": self._queue.qsize(),
            "written": self._written,
            "dropped": self._dropped,
        }


# ── module-level singleton ────────────────────────────────────────────────────

_recorder: Optional[UsageRecorder] = None


def get_usage_recorder() -> UsageRecorder:
    global _recorder
    if _recorder is None:
        _recorder = UsageRecorder()
    return _recorder


def record(
    *,
    agent: str,
    model: str,
    provider: str = "gemini",
    user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    image_count: int = 0,
    audio_input_tokens: int = 0,
    latency_ms: Optional[int] = None,
    status: str = "success",
    error_message: Optional[str] = None,
) -> None:
    """Module-level convenience wrapper around the singleton recorder."""
    get_usage_recorder().record(
        agent=agent, model=model, provider=provider, user_id=user_id,
        guest_session_id=guest_session_id, input_tokens=input_tokens,
        output_tokens=output_tokens, image_count=image_count,
        audio_input_tokens=audio_input_tokens, latency_ms=latency_ms,
        status=status, error_message=error_message,
    )


def heartbeat(*, user_id: Optional[int] = None, guest_session_id: Optional[str] = None) -> None:
    get_usage_recorder().heartbeat(user_id=user_id, guest_session_id=guest_session_id)
