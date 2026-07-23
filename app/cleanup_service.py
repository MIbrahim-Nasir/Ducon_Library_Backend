"""Shared retention cleanup used by POST /guest/cleanup and the in-process weekly scheduler.

Deletes only data that is past its expiry / terminal state **and** older than
``CLEANUP_MIN_AGE_DAYS`` (default 7) where age applies. Stale in-progress job
mark-failed uses a short hours threshold (not deletion).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app import storage
from app.admin.settings_store import cfg_int
from app.db.models import EmailOtp, GuestGeneration, RevokedJti
from app.designer_cleanup import cleanup_designer_jobs

logger = logging.getLogger(__name__)


def min_age_days() -> int:
    return max(1, cfg_int("CLEANUP_MIN_AGE_DAYS", 7))


async def run_cleanup(db: AsyncSession, *, commit: bool = True) -> dict[str, int]:
    """Purge expired guest gens, old designer jobs, stale job rows, OTPs, JTIs.

    ``commit=False`` lets the weekly scheduler wrap work in a single advisory-lock
    transaction.
    """
    now = datetime.now(timezone.utc)
    age_cutoff = now - timedelta(days=min_age_days())

    # Unclaimed guest generations: must be expired AND older than min age.
    result = await db.execute(
        select(GuestGeneration).where(
            GuestGeneration.expires_at.is_not(None),
            GuestGeneration.expires_at < now,
            GuestGeneration.user_id.is_(None),
            GuestGeneration.generated_at < age_cutoff,
        )
    )
    expired = result.scalars().all()

    for gen in expired:
        try:
            storage.delete_guest_generation(gen.url)
        except Exception:
            logger.debug("guest file delete failed for %s", gen.url, exc_info=True)

    guest_deleted = 0
    if expired:
        await db.execute(
            delete(GuestGeneration).where(
                GuestGeneration.id.in_([g.id for g in expired])
            )
        )
        guest_deleted = len(expired)

    designer_stats = await cleanup_designer_jobs(db)

    # Generation jobs stranded by a worker restart mid-run — mark failed, do not delete.
    stale_gen = await db.execute(
        text(
            "UPDATE generation_jobs "
            "SET status = 'failed', error = 'stale: worker restarted mid-run', "
            "    updated_at = NOW() "
            "WHERE status IN ('queued', 'running') "
            "  AND updated_at < NOW() - INTERVAL '2 hours'"
        )
    )
    generation_jobs_marked_stale = stale_gen.rowcount or 0

    # Expired JWT revocations and OTPs — safe once past expires_at.
    jti_result = await db.execute(
        delete(RevokedJti).where(RevokedJti.expires_at < now)
    )
    revoked_jtis_deleted = jti_result.rowcount or 0

    otp_result = await db.execute(
        delete(EmailOtp).where(EmailOtp.expires_at < now)
    )
    email_otps_deleted = otp_result.rowcount or 0

    stats = {
        "guest_generations_deleted": guest_deleted,
        "generation_jobs_marked_stale": generation_jobs_marked_stale,
        "revoked_jtis_deleted": revoked_jtis_deleted,
        "email_otps_deleted": email_otps_deleted,
        **designer_stats,
    }

    if commit and any(stats.values()):
        await db.commit()

    return stats
