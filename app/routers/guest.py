"""
app/routers/guest.py
────────────────────
Guest-user endpoints:

  GET  /guest/usage                     — { generations, chat, voice } used/limit
  GET  /guest/gen-count                 — legacy { used, limit } for generations
  POST /auth/claim-guest-generations    — migrates guest generations to an authenticated user
  POST /guest/cleanup                   — deletes expired, unclaimed records (run via cron)
  GET  /guest/generations/{id}/image    — serves a local guest generation (local mode only)

Rate limiting:
  - Cloudflare Turnstile on guest generation and chat requests.
  - Per-feature session limits (generation / chat / voice).
  - IP total cap across all features (GUEST_IP_TOTAL_LIMIT).

Generation limits count one completed output image per pipeline run — internal
eval/retry rounds do not increment the counter.
"""

import hmac
import os
import uuid

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import storage
from app.auth import get_current_user
from app.config import IS_PRODUCTION
from app.db.database import get_db
from app.hashing import sha256_hex
from app.rate_limiter import require_rate_limit
from app.signed_urls import verify_guest_generation
from app.db.models import Generation, GuestConsentAudit, GuestGeneration, GuestSession
from app.guest_usage import (
    GuestUsageKind,
    enforce_guest_limit,
    increment_guest_usage,
    usage_snapshot,
)

router = APIRouter(tags=["guest"])

TURNSTILE_SECRET     = os.getenv("TURNSTILE_SECRET_KEY", "")
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
# Shared secret required to run the cleanup job (set in the cron environment).
CLEANUP_CRON_SECRET  = os.getenv("GUEST_CLEANUP_SECRET", "")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hash(value: str) -> str:
    return sha256_hex(value)


async def verify_turnstile(token: str, ip: str) -> bool:
    """
    Verifies a Cloudflare Turnstile token server-to-server.

    - If TURNSTILE_SECRET_KEY is not set:
        • production → fail closed (verification cannot be trusted).
        • local dev  → skip verification.
    - If TURNSTILE_SECRET_KEY is set: token MUST be present and valid.
    """
    if not TURNSTILE_SECRET:
        # Fail closed in production: a missing secret must not silently disable
        # bot protection on paid AI endpoints.
        return not IS_PRODUCTION
    if not token:
        return False
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(
            TURNSTILE_VERIFY_URL,
            data={"secret": TURNSTILE_SECRET, "response": token, "remoteip": ip},
        )
    return r.json().get("success", False)


async def get_or_create_guest_session(
    db: AsyncSession,
    session_id: str,
    ip: str,
    *,
    usage_kind: GuestUsageKind = GuestUsageKind.GENERATION,
) -> GuestSession:
    """
    Fetch/create guest session and enforce limits for the requested feature.
    Used by generation, chat, and voice entry points.
    """
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid X-Guest-Session-Id. Must be a valid UUID.")
    return await enforce_guest_limit(db, session_id, _hash(ip), usage_kind)


async def increment_guest_count(db: AsyncSession, session: GuestSession) -> None:
    """Increment generation counter after a saved output image (legacy name)."""
    await increment_guest_usage(db, session, GuestUsageKind.GENERATION)


async def log_guest_consent(
    db: AsyncSession,
    session_id: str,
    ip: str,
    event: str = "ai_generation_guest_consent",
) -> None:
    """Append an audit record — hashed identifiers only, no raw PII."""
    audit = GuestConsentAudit(
        session_id_hash=_hash(session_id),
        ip_hash=_hash(ip),
        event=event,
    )
    db.add(audit)
    await db.flush()


# ── GET /guest/usage ───────────────────────────────────────────────────────────

@router.get("/guest/usage")
async def guest_usage(
    x_guest_session_id: str = Header(..., alias="X-Guest-Session-Id"),
    db: AsyncSession = Depends(get_db),
):
    """Returns per-feature usage for a guest session."""
    result = await db.execute(
        select(GuestSession).where(GuestSession.session_id == x_guest_session_id)
    )
    session = result.scalar_one_or_none()
    return usage_snapshot(session)


# ── GET /guest/gen-count (legacy) ──────────────────────────────────────────────

@router.get("/guest/gen-count")
async def guest_gen_count(
    x_guest_session_id: str = Header(..., alias="X-Guest-Session-Id"),
    db: AsyncSession = Depends(get_db),
):
    """Legacy endpoint — generation counts only."""
    result = await db.execute(
        select(GuestSession).where(GuestSession.session_id == x_guest_session_id)
    )
    session = result.scalar_one_or_none()
    snap = usage_snapshot(session)
    return {"used": snap["generations"]["used"], "limit": snap["generations"]["limit"]}


# ── POST /auth/claim-guest-generations ────────────────────────────────────────

class ClaimRequest(BaseModel):
    guest_session_id: str


@router.post("/auth/claim-guest-generations")
async def claim_guest_generations(
    body: ClaimRequest,
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Migrates unexpired guest generations to the authenticated user's account.
    """
    from datetime import datetime, timezone

    # Defense-in-depth: session ids are random UUIDs, but rate-limit anyway so a
    # client can't brute-force-scan guest sessions to claim someone else's work.
    require_rate_limit(request, max_requests=20, window_seconds=60, key_prefix="claim_guest")

    now = datetime.now(timezone.utc)

    result = await db.execute(
        select(GuestGeneration).where(
            GuestGeneration.guest_session_id == body.guest_session_id,
            GuestGeneration.user_id.is_(None),
            GuestGeneration.expires_at > now,
        )
    )
    guest_generations = result.scalars().all()

    claimed_ids = []
    for gen in guest_generations:
        try:
            new_key = storage.move_guest_to_user(
                session_id=body.guest_session_id,
                filename=gen.generation_name,
                user_id=current_user.id,
            )
        except Exception:
            continue

        db.add(Generation(
            user_id=current_user.id,
            generation_name=gen.generation_name,
            url=new_key,
            ducon_image_id=gen.ducon_image_id,
        ))
        claimed_ids.append(gen.id)

    if claimed_ids:
        await db.execute(
            delete(GuestGeneration).where(GuestGeneration.id.in_(claimed_ids))
        )

    await db.commit()
    return {"claimed": len(claimed_ids)}


# ── POST /guest/cleanup ────────────────────────────────────────────────────────

async def _run_cleanup(db: AsyncSession) -> int:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    result = await db.execute(
        select(GuestGeneration).where(
            GuestGeneration.expires_at < now,
            GuestGeneration.user_id.is_(None),
        )
    )
    expired = result.scalars().all()

    for gen in expired:
        try:
            storage.delete_guest_generation(gen.url)
        except Exception:
            pass

    if expired:
        await db.execute(
            delete(GuestGeneration).where(
                GuestGeneration.id.in_([g.id for g in expired])
            )
        )
        await db.commit()

    return len(expired)


@router.post("/guest/cleanup")
async def guest_cleanup(
    x_cron_secret: str = Header("", alias="X-Cron-Secret"),
    db: AsyncSession = Depends(get_db),
):
    # Protect the destructive cleanup job. When a secret is configured it must
    # match; in production a secret is mandatory.
    if CLEANUP_CRON_SECRET:
        if not hmac.compare_digest(x_cron_secret or "", CLEANUP_CRON_SECRET):
            raise HTTPException(status_code=403, detail="Forbidden.")
    elif IS_PRODUCTION:
        raise HTTPException(status_code=403, detail="Cleanup secret not configured.")

    deleted = await _run_cleanup(db)
    return {"deleted": deleted}


# ── GET /guest/generations/{id}/image — local mode only ───────────────────────

@router.get("/guest/generations/{generation_id}/image")
async def serve_guest_generation(
    generation_id: int,
    token: str = "",
    db: AsyncSession = Depends(get_db),
):
    # Signed-URL check: prevents enumerating other guests' generations by id.
    if not verify_guest_generation(generation_id, token):
        raise HTTPException(status_code=403, detail="Invalid or missing signature.")

    result = await db.execute(
        select(GuestGeneration).where(GuestGeneration.id == generation_id)
    )
    gen = result.scalar_one_or_none()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    path = storage.serve_local_path(gen.url)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Generation file not found")

    return FileResponse(path, media_type="image/png")
