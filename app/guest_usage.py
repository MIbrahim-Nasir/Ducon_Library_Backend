"""
Guest usage limits — one counter per feature, enforced per device identity.

Identity resolution (most to least stable):
  1. fingerprint_hash (X-Guest-Fingerprint header) — browser/hardware fingerprint computed
     client-side and sent on every request.  When the server sees a known fingerprint it
     returns the *existing* session row regardless of the UUID in X-Guest-Session-Id, so
     clearing localStorage, going incognito, or sending a fresh UUID does NOT reset quota
     as long as the device signals stay the same.
  2. session_id (X-Guest-Session-Id header or signed guest cookie) — per-session UUID
     lookup when no fingerprint is available.

composite_hash / subnet_key / ja4 / asn are stored on the row for audit and optional
shared caps but are NOT used to merge different session UUIDs into one quota row.

Each device gets independent limits; coworkers on the same office NAT are unaffected by
each other's usage (no shared IP or subnet pool by default).

Optional ``GUEST_IP_TOTAL_LIMIT`` / ``GUEST_SUBNET_LIMIT`` (default 0 = disabled) can
re-enable shared caps for heavy public deployments; leave at 0 for office / corporate NAT.

Units counted per completed user-facing action:
  • generation — one saved image output (eval/retry rounds inside the pipeline do not count)
  • chat       — one completed chat turn (/chat/message through SSE done, not tool_result continuations)
  • voice       — one completed voice turn (user and/or model transcript committed)
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.settings_store import cfg
from app.db.models import GuestSession
from app.error_logger import log_warning
from app.hashing import sha256_hex

# Accepted fingerprint: 64-char lowercase hex (SHA-256 produced by the frontend script).
_FP_RE = re.compile(r"^[0-9a-f]{64}$")

GUEST_GEN_LIMIT = 3
GUEST_CHAT_LIMIT = 10
GUEST_VOICE_LIMIT = 5
GUEST_IP_TOTAL_LIMIT = 0
GUEST_SUBNET_LIMIT = 0


def _guest_gen_limit() -> int:
    return int(cfg("GUEST_GEN_LIMIT", GUEST_GEN_LIMIT))


def _guest_chat_limit() -> int:
    return int(cfg("GUEST_CHAT_LIMIT", GUEST_CHAT_LIMIT))


def _guest_voice_limit() -> int:
    return int(cfg("GUEST_VOICE_LIMIT", GUEST_VOICE_LIMIT))


def _guest_ip_total_limit() -> int:
    return int(cfg("GUEST_IP_TOTAL_LIMIT", GUEST_IP_TOTAL_LIMIT))


def _guest_subnet_limit() -> int:
    return int(cfg("GUEST_SUBNET_LIMIT", GUEST_SUBNET_LIMIT))

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
        return _guest_gen_limit()
    if kind == GuestUsageKind.CHAT:
        return _guest_chat_limit()
    return _guest_voice_limit()


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


def _validate_fingerprint(fp: Optional[str]) -> Optional[str]:
    """Return the fingerprint if it looks like a valid SHA-256 hex, else None."""
    if fp and _FP_RE.match(fp):
        return fp
    return None


def normalise_fingerprint_hash(raw: Optional[str]) -> Optional[str]:
    """Validate client fingerprint header and return the stored server-side hash."""
    if not raw or not raw.strip():
        return None
    validated = _validate_fingerprint(raw.strip())
    if not validated:
        return None
    return sha256_hex(validated)


async def _subnet_total_usage(db: AsyncSession, subnet: str) -> int:
    result = await db.execute(
        select(
            func.coalesce(func.sum(GuestSession.generation_count), 0)
            + func.coalesce(func.sum(GuestSession.chat_turn_count), 0)
            + func.coalesce(func.sum(GuestSession.voice_turn_count), 0)
        ).where(GuestSession.subnet_key == subnet)
    )
    return int(result.scalar() or 0)


def _bind_identity_fields(
    session: GuestSession,
    *,
    ip_hash: str,
    fingerprint_hash: Optional[str],
    composite_hash: Optional[str],
    subnet_key: Optional[str],
    ja4_fingerprint: Optional[str],
    asn: Optional[str],
) -> dict:
    """Return column updates to persist on an existing guest session row."""
    updates: dict = {}
    if ip_hash and session.ip_hash != ip_hash:
        updates["ip_hash"] = ip_hash
    if fingerprint_hash and not session.fingerprint_hash:
        updates["fingerprint_hash"] = fingerprint_hash
    if composite_hash and not session.composite_hash:
        updates["composite_hash"] = composite_hash
    if subnet_key and not session.subnet_key:
        updates["subnet_key"] = subnet_key
    if ja4_fingerprint and not session.ja4_fingerprint:
        updates["ja4_fingerprint"] = ja4_fingerprint
    if asn and not session.asn:
        updates["asn"] = asn
    return updates


async def get_guest_session_row(
    db: AsyncSession,
    session_id: str,
    ip_hash: str,
    *,
    fingerprint_hash: Optional[str] = None,
    composite_hash: Optional[str] = None,
    subnet_key: Optional[str] = None,
    ja4_fingerprint: Optional[str] = None,
    asn: Optional[str] = None,
    create: bool = True,
) -> Optional[GuestSession]:
    """Fetch a guest session row; create one when ``create`` is True (default).

    Read-only callers (``GET /guest/usage``, ``GET /guest/gen-count``) should pass
    ``create=False`` so a poll does not mint a quota row.

    Identity resolution:
      • If ``fingerprint_hash`` is provided and a session already has that
        fingerprint, that existing session is returned (UUID rotation ignored).
      • Otherwise the row is looked up by ``session_id``, creating one if absent
        and ``create`` is True.
      • ``composite_hash`` and related signals are persisted for audit only — they
        never merge different UUIDs into a shared quota row.
    """
    # fingerprint_hash arrives pre-normalised (double-hashed) from the router layer.

    # ── 1. Fingerprint-first lookup ──────────────────────────────────────────
    if fingerprint_hash:
        fp_result = await db.execute(
            select(GuestSession).where(GuestSession.fingerprint_hash == fingerprint_hash)
        )
        existing = fp_result.scalar_one_or_none()
        if existing is not None:
            updates = _bind_identity_fields(
                existing,
                ip_hash=ip_hash,
                fingerprint_hash=fingerprint_hash,
                composite_hash=composite_hash,
                subnet_key=subnet_key,
                ja4_fingerprint=ja4_fingerprint,
                asn=asn,
            )
            if updates:
                await db.execute(
                    update(GuestSession)
                    .where(GuestSession.id == existing.id)
                    .values(**updates)
                )
                for key, value in updates.items():
                    setattr(existing, key, value)
                await db.flush()
            return existing

    # ── 2. UUID-based lookup / create ────────────────────────────────────────
    result = await db.execute(
        select(GuestSession).where(GuestSession.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        if not create:
            return None
        session = GuestSession(
            session_id=session_id,
            ip_hash=ip_hash,
            fingerprint_hash=fingerprint_hash,
            composite_hash=composite_hash,
            subnet_key=subnet_key,
            ja4_fingerprint=ja4_fingerprint,
            asn=asn,
            generation_count=0,
            chat_turn_count=0,
            voice_turn_count=0,
        )
        db.add(session)
        await db.flush()
    else:
        updates = _bind_identity_fields(
            session,
            ip_hash=ip_hash,
            fingerprint_hash=fingerprint_hash,
            composite_hash=composite_hash,
            subnet_key=subnet_key,
            ja4_fingerprint=ja4_fingerprint,
            asn=asn,
        )
        if updates:
            await db.execute(
                update(GuestSession)
                .where(GuestSession.id == session.id)
                .values(**updates)
            )
            for key, value in updates.items():
                setattr(session, key, value)
            await db.flush()

    return session


async def enforce_guest_limit(
    db: AsyncSession,
    session_id: str,
    ip_hash: str,
    kind: GuestUsageKind,
    *,
    fingerprint_hash: Optional[str] = None,
    composite_hash: Optional[str] = None,
    subnet_key: Optional[str] = None,
    ja4_fingerprint: Optional[str] = None,
    asn: Optional[str] = None,
) -> GuestSession:
    """
    Ensure the guest may start another unit of this kind.

    Uses fingerprint-first identity: same device always maps to the same session
    row even after localStorage is cleared or a new UUID is sent.

    Raises HTTP 429 when this session's feature limit is exhausted, or when an
    optional shared IP / subnet cap is enabled and exhausted.
    """
    ip_cap = _guest_ip_total_limit()
    if ip_cap > 0:
        ip_total = await _ip_total_usage(db, ip_hash)
        if ip_total >= ip_cap:
            await log_warning(
                "guest",
                "guest_usage.enforce_guest_limit",
                "Guest IP total limit exceeded",
                guest_session_id=session_id,
                endpoint=f"guest.{kind.value}",
                http_status=429,
                extra={"kind": kind.value, "ip_total": ip_total, "limit": ip_cap},
            )
            raise HTTPException(status_code=429, detail=_LIMIT_EXCEEDED)

    subnet_cap = _guest_subnet_limit()
    if subnet_cap > 0 and subnet_key:
        subnet_total = await _subnet_total_usage(db, subnet_key)
        if subnet_total >= subnet_cap:
            await log_warning(
                "guest",
                "guest_usage.enforce_guest_limit",
                "Guest subnet total limit exceeded",
                guest_session_id=session_id,
                endpoint=f"guest.{kind.value}",
                http_status=429,
                extra={
                    "kind": kind.value,
                    "subnet_total": subnet_total,
                    "limit": subnet_cap,
                    "subnet_key": subnet_key,
                },
            )
            raise HTTPException(status_code=429, detail=_LIMIT_EXCEEDED)

    session = await get_guest_session_row(
        db,
        session_id,
        ip_hash,
        fingerprint_hash=fingerprint_hash,
        composite_hash=composite_hash,
        subnet_key=subnet_key,
        ja4_fingerprint=ja4_fingerprint,
        asn=asn,
    )
    limit = _limit_for(kind)
    current = getattr(session, _count_column(kind), 0) or 0
    if current >= limit:
        await log_warning(
            "guest",
            "guest_usage.enforce_guest_limit",
            f"Guest {kind.value} limit exceeded",
            guest_session_id=session_id,
            endpoint=f"guest.{kind.value}",
            http_status=429,
            extra={"kind": kind.value, "used": current, "limit": limit},
        )
        raise HTTPException(status_code=429, detail=_LIMIT_EXCEEDED)
    return session


async def increment_guest_usage(
    db: AsyncSession,
    session: GuestSession,
    kind: GuestUsageKind,
) -> bool:
    """Increment after a successful user-facing unit (not internal retries).

    Uses a conditional DB-side ``UPDATE … WHERE count < limit`` so concurrent
    requests that both passed ``enforce_guest_limit`` cannot push the counter
    past the configured cap (closes the enforce→increment TOCTOU).

    Returns True if the increment was applied, False if the session was already
    at/over the limit (caller should treat as soft overage — generation may
    already be persisted).
    """
    from datetime import datetime, timezone

    col = _count_column(kind)
    col_attr = getattr(GuestSession, col)
    limit = _limit_for(kind)
    result = await db.execute(
        update(GuestSession)
        .where(GuestSession.id == session.id, col_attr < limit)
        .values(**{col: col_attr + 1, "last_used_at": datetime.now(timezone.utc)})
    )
    await db.flush()
    applied = bool(getattr(result, "rowcount", 0))
    # Keep the in-memory ORM object consistent with the DB for any later reads.
    try:
        await db.refresh(session)
    except Exception:
        pass
    return applied


def usage_snapshot(session: Optional[GuestSession]) -> dict:
    gen_used = int(session.generation_count or 0) if session else 0
    chat_used = int(session.chat_turn_count or 0) if session else 0
    voice_used = int(session.voice_turn_count or 0) if session else 0
    return {
        "generations": {"used": gen_used, "limit": _guest_gen_limit()},
        "chat": {"used": chat_used, "limit": _guest_chat_limit()},
        "voice": {"used": voice_used, "limit": _guest_voice_limit()},
        # Legacy field for older clients
        "used": gen_used,
        "limit": _guest_gen_limit(),
    }
