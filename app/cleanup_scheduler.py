"""In-process weekly cleanup scheduler with Postgres single-flight locking.

Every gunicorn worker starts this loop; only one acquires ``pg_try_advisory_xact_lock``
per tick. Last-run is persisted in ``app_settings`` so restarts do not re-run early.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.settings_store import cfg_bool, cfg_int
from app.cleanup_service import run_cleanup
from app.db.models import AppSetting

logger = logging.getLogger(__name__)

# Distinct from schema create_all lock (788214001) in main.lifespan.
_CLEANUP_ADVISORY_LOCK_KEY = 788214002
_LAST_RUN_NS = "system"
_LAST_RUN_KEY = "CLEANUP_LAST_RUN_AT"
_STARTUP_DELAY_SECONDS = 60
_CHECK_PERIOD_SECONDS = 3600  # re-evaluate due status hourly


def cleanup_enabled() -> bool:
    return cfg_bool("CLEANUP_ENABLED", True)


def cleanup_interval_days() -> int:
    return max(1, cfg_int("CLEANUP_INTERVAL_DAYS", 7))


async def _get_last_run(db: AsyncSession) -> datetime | None:
    result = await db.execute(
        select(AppSetting).where(
            AppSetting.namespace == _LAST_RUN_NS,
            AppSetting.key == _LAST_RUN_KEY,
        )
    )
    row = result.scalar_one_or_none()
    if not row or not row.value:
        return None
    raw = row.value.strip().strip('"')
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


async def _set_last_run(db: AsyncSession, when: datetime) -> None:
    iso = when.astimezone(timezone.utc).isoformat()
    result = await db.execute(
        select(AppSetting).where(
            AppSetting.namespace == _LAST_RUN_NS,
            AppSetting.key == _LAST_RUN_KEY,
        )
    )
    row = result.scalar_one_or_none()
    if row:
        row.value = iso
        row.updated_at = when
    else:
        db.add(
            AppSetting(
                namespace=_LAST_RUN_NS,
                key=_LAST_RUN_KEY,
                value=iso,
                value_type="string",
                is_secret=False,
                description="ISO timestamp of last successful scheduled cleanup run",
            )
        )


async def run_scheduled_cleanup_tick() -> dict:
    """Attempt one scheduled cleanup under an advisory lock.

    Returns a small status dict for logging/tests. Idempotent across workers.
    """
    if not cleanup_enabled():
        logger.debug("cleanup: disabled (CLEANUP_ENABLED=false)")
        return {"status": "disabled"}

    from app.db.database import async_session_maker

    now = datetime.now(timezone.utc)
    interval = timedelta(days=cleanup_interval_days())

    async with async_session_maker() as db:
        async with db.begin():
            locked = await db.scalar(
                text(f"SELECT pg_try_advisory_xact_lock({_CLEANUP_ADVISORY_LOCK_KEY})")
            )
            if not locked:
                logger.info("cleanup: skipped — another worker holds the advisory lock")
                return {"status": "lock_skipped"}

            last = await _get_last_run(db)
            if last is not None and (now - last) < interval:
                logger.debug(
                    "cleanup: not due yet (last_run=%s, interval_days=%s)",
                    last.isoformat(),
                    cleanup_interval_days(),
                )
                return {"status": "not_due", "last_run": last.isoformat()}

            logger.info(
                "cleanup: starting (last_run=%s, min_age_days=%s, interval_days=%s)",
                last.isoformat() if last else "never",
                cfg_int("CLEANUP_MIN_AGE_DAYS", 7),
                cleanup_interval_days(),
            )
            stats = await run_cleanup(db, commit=False)
            await _set_last_run(db, now)
            logger.info("cleanup: finished %s", stats)
            return {"status": "ran", "stats": stats}


async def weekly_cleanup_loop() -> None:
    """Background task: grace delay, then hourly due-checks with weekly retention."""
    await asyncio.sleep(_STARTUP_DELAY_SECONDS)
    while True:
        try:
            await run_scheduled_cleanup_tick()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("cleanup: scheduled tick failed")
        await asyncio.sleep(_CHECK_PERIOD_SECONDS)
