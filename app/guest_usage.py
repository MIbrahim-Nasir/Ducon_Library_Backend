"""
Guest usage limits — one counter per feature, enforced per completed user-facing unit:
  • generation — one saved image output (eval/retry rounds inside the pipeline do not count)
  • chat       — one completed chat turn (/chat/message through SSE done, not tool_result continuations)
  • voice      — one completed voice turn (user and/or model transcript committed)
"""
from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import GuestSession

GUEST_GEN_LIMIT = int(os.getenv("GUEST_GEN_LIMIT", "3"))
GUEST_CHAT_LIMIT = int(os.getenv("GUEST_CHAT_LIMIT", "10"))
GUEST_VOICE_LIMIT = int(os.getenv("GUEST_VOICE_LIMIT", "5"))
# Total usage across all guest sessions from one IP (generations + chat + voice).
GUEST_IP_TOTAL_LIMIT = int(os.getenv("GUEST_IP_TOTAL_LIMIT", os.getenv("GUEST_IP_GEN_LIMIT", "15")))

_LIMIT_EXCEEDED = {
    "detail": "Guest limit reached. Sign up to continue.",
    "code": "GUEST_LIMIT_REACHED",
}


class GuestUsageKind(str, Enum):
    GENERATION = "generation"
    CHAT = "chat"
    VOICE = "voice"


def _limit_for(kind: GuestUsageKind) -> int:
    if kind == GuestUsageKind.GENERATION:
        return GUEST_GEN_LIMIT
    if kind == GuestUsageKind.CHAT:
        return GUEST_CHAT_LIMIT
    return GUEST_VOICE_LIMIT


def _count_column(kind: GuestUsageKind) -> str:
    if kind == GuestUsageKind.GENERATION:
        return "generation_count"
    if kind == GuestUsageKind.CHAT:
        return "chat_turn_count"
    return "voice_turn_count"


def _session_total_usage(session: GuestSession) -> int:
    return (
        int(session.generation_count or 0)
        + int(session.chat_turn_count or 0)
        + int(session.voice_turn_count or 0)
    )


async def _ip_total_usage(db: AsyncSession, ip_hash: str) -> int:
    result = await db.execute(
        select(
            func.coalesce(func.sum(GuestSession.generation_count), 0)
            + func.coalesce(func.sum(GuestSession.chat_turn_count), 0)
            + func.coalesce(func.sum(GuestSession.voice_turn_count), 0)
        ).where(GuestSession.ip_hash == ip_hash)
    )
    return int(result.scalar() or 0)


async def get_guest_session_row(
    db: AsyncSession,
    session_id: str,
    ip_hash: str,
) -> GuestSession:
    """Fetch or create a guest session row (does not enforce limits)."""
    result = await db.execute(
        select(GuestSession).where(GuestSession.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        session = GuestSession(
            session_id=session_id,
            ip_hash=ip_hash,
            generation_count=0,
            chat_turn_count=0,
            voice_turn_count=0,
        )
        db.add(session)
        await db.flush()
    return session


async def enforce_guest_limit(
    db: AsyncSession,
    session_id: str,
    ip_hash: str,
    kind: GuestUsageKind,
) -> GuestSession:
    """
    Ensure the guest may start another unit of this kind.
    Raises HTTP 429 when session or IP totals are exhausted.
    """
    ip_total = await _ip_total_usage(db, ip_hash)
    if ip_total >= GUEST_IP_TOTAL_LIMIT:
        raise HTTPException(status_code=429, detail=_LIMIT_EXCEEDED)

    session = await get_guest_session_row(db, session_id, ip_hash)
    limit = _limit_for(kind)
    current = getattr(session, _count_column(kind), 0) or 0
    if current >= limit:
        raise HTTPException(status_code=429, detail=_LIMIT_EXCEEDED)
    return session


async def increment_guest_usage(
    db: AsyncSession,
    session: GuestSession,
    kind: GuestUsageKind,
) -> None:
    """Increment after a successful user-facing unit (not internal retries).

    Uses a DB-side ``col = col + 1`` expression so concurrent requests can't
    clobber each other's increment (avoids the lost-update race a read →
    modify → write in Python would have).
    """
    from datetime import datetime, timezone

    col = _count_column(kind)
    col_attr = getattr(GuestSession, col)
    await db.execute(
        update(GuestSession)
        .where(GuestSession.id == session.id)
        .values(**{col: col_attr + 1, "last_used_at": datetime.now(timezone.utc)})
    )
    await db.flush()
    # Keep the in-memory ORM object consistent with the DB for any later reads.
    # Best-effort: the authoritative increment already happened in the DB above;
    # a refresh failure (e.g. object loaded on another session) must not fail the
    # request.
    try:
        await db.refresh(session)
    except Exception:
        pass


def usage_snapshot(session: Optional[GuestSession]) -> dict:
    gen_used = int(session.generation_count or 0) if session else 0
    chat_used = int(session.chat_turn_count or 0) if session else 0
    voice_used = int(session.voice_turn_count or 0) if session else 0
    return {
        "generations": {"used": gen_used, "limit": GUEST_GEN_LIMIT},
        "chat": {"used": chat_used, "limit": GUEST_CHAT_LIMIT},
        "voice": {"used": voice_used, "limit": GUEST_VOICE_LIMIT},
        # Legacy field for older clients
        "used": gen_used,
        "limit": GUEST_GEN_LIMIT,
    }
