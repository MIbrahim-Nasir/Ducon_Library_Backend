"""Retention cleanup for persisted designer job rows and in-memory registry."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DesignerJobRow

_TERMINAL = frozenset({"completed", "failed", "cancelled"})


def _retention_days() -> int:
    # Default 7 days — aligned with in-process weekly cleanup (CLEANUP_MIN_AGE_DAYS).
    return max(1, int(os.getenv("DESIGNER_JOB_RETENTION_DAYS", "7")))


def _stale_running_hours() -> int:
    return max(1, int(os.getenv("DESIGNER_JOB_STALE_RUNNING_HOURS", "2")))


async def cleanup_designer_jobs(db: AsyncSession) -> dict[str, int]:
    """Prune old terminal job rows and mark crashed in-flight jobs as failed.

    Does not delete user ``generations`` — only the job metadata/event log in
    ``designer_jobs``. Run from ``POST /guest/cleanup`` and the weekly in-process scheduler.
    """
    now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(hours=_stale_running_hours())
    retention_cutoff = now - timedelta(days=_retention_days())

    stale = await db.execute(
        select(DesignerJobRow.id).where(
            DesignerJobRow.status.notin_(tuple(_TERMINAL)),
            DesignerJobRow.updated_at < stale_cutoff,
        )
    )
    stale_ids = [row[0] for row in stale.all()]
    marked_stale = 0
    if stale_ids:
        await db.execute(
            update(DesignerJobRow)
            .where(DesignerJobRow.id.in_(stale_ids))
            .values(
                status="failed",
                error="Job timed out (worker may have restarted).",
                updated_at=now,
            )
        )
        marked_stale = len(stale_ids)

    old = await db.execute(
        select(DesignerJobRow.id).where(
            DesignerJobRow.status.in_(tuple(_TERMINAL)),
            DesignerJobRow.updated_at < retention_cutoff,
        )
    )
    old_ids = [row[0] for row in old.all()]
    deleted = 0
    if old_ids:
        await db.execute(delete(DesignerJobRow).where(DesignerJobRow.id.in_(old_ids)))
        deleted = len(old_ids)

    if marked_stale or deleted:
        pass  # caller commits

    # Lazy import: avoid pulling designer_agent into cleanup/scheduler import graph.
    if old_ids:
        from app.designer_agent import JOBS

        for job_id in old_ids:
            JOBS.pop(job_id, None)

    return {"designer_jobs_marked_stale": marked_stale, "designer_jobs_deleted": deleted}
