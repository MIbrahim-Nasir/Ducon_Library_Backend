"""
app/routers/guest.py
────────────────────
Guest-user endpoints:

  GET  /guest/gen-count                 — returns { used, limit } for a guest session
  POST /auth/claim-guest-generations    — migrates guest generations to an authenticated user
  POST /guest/cleanup                   — deletes expired, unclaimed records (run via cron)
  GET  /guest/generations/{id}/image    — serves a local guest generation (local mode only)

Rate limiting strategy:
  - Cloudflare Turnstile token required on every guest generation request.
    The token is verified server-to-server — bots and scripts cannot generate
    valid tokens. This replaces the previous IP-based rate limiting.
  - Session-based limit (GUEST_GEN_LIMIT) is still enforced as a business rule:
    each guest session gets N free generations before being asked to sign up.
"""

import hashlib
import os
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import storage
from app.auth import get_current_user
from app.db.database import get_db
from app.db.models import Generation, GuestConsentAudit, GuestGeneration, GuestSession

router = APIRouter(tags=["guest"])

GUEST_GEN_LIMIT      = int(os.getenv("GUEST_GEN_LIMIT",    "3"))
# Higher than session limit to allow a few legitimate session resets (cleared cookies,
# different browser) while still blocking the most obvious abuse.
GUEST_IP_GEN_LIMIT   = int(os.getenv("GUEST_IP_GEN_LIMIT", "6"))
TURNSTILE_SECRET     = os.getenv("TURNSTILE_SECRET_KEY", "")
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


_LIMIT_EXCEEDED = {
    "detail": "Guest generation limit reached. Sign up to continue.",
    "code": "GUEST_LIMIT_REACHED",
}


async def verify_turnstile(token: str, ip: str) -> bool:
    """
    Verifies a Cloudflare Turnstile token server-to-server.

    - If TURNSTILE_SECRET_KEY is not set: skips verification (local dev only).
    - If TURNSTILE_SECRET_KEY is set: token MUST be present and valid.
      A missing or empty token is an immediate failure — the frontend must
      always include the token when a site key is configured.
    """
    if not TURNSTILE_SECRET:
        return True  # Dev mode — no secret configured, skip
    if not token:
        return False  # Secret is configured but token was not sent — reject
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
) -> GuestSession:
    """
    Fetch or create the GuestSession row, enforcing two limits:

    1. IP limit (primary gate) — total generations across ALL sessions from this IP.
       Stops the most common bypass: clearing cookies to get a fresh session ID.
       Turnstile stops bots; IP limit stops humans who clear site data.
       Set GUEST_IP_GEN_LIMIT slightly above GUEST_GEN_LIMIT so a legitimate user
       who accidentally clears cookies still gets a couple of retries.

    2. Session limit (secondary) — generations on this specific session ID.
       This is the counter the frontend UI shows to the user.
    """
    ip_hash = _hash(ip)

    # ── Primary gate: IP total (single aggregate query, not fetching all rows) ──
    ip_result = await db.execute(
        select(func.coalesce(func.sum(GuestSession.generation_count), 0))
        .where(GuestSession.ip_hash == ip_hash)
    )
    ip_total = ip_result.scalar()
    if ip_total >= GUEST_IP_GEN_LIMIT:
        raise HTTPException(status_code=429, detail=_LIMIT_EXCEEDED)

    # ── Session lookup / creation ─────────────────────────────────────────────
    result = await db.execute(
        select(GuestSession).where(GuestSession.session_id == session_id)
    )
    session = result.scalar_one_or_none()

    if session is None:
        session = GuestSession(session_id=session_id, ip_hash=ip_hash, generation_count=0)
        db.add(session)
        await db.flush()
    elif session.generation_count >= GUEST_GEN_LIMIT:
        raise HTTPException(status_code=429, detail=_LIMIT_EXCEEDED)

    return session


async def increment_guest_count(db: AsyncSession, session: GuestSession) -> None:
    """Increment the generation counter and update last_used_at."""
    session.generation_count += 1
    session.last_used_at = datetime.now(timezone.utc)
    await db.flush()


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


# ── GET /guest/gen-count ───────────────────────────────────────────────────────

@router.get("/guest/gen-count")
async def guest_gen_count(
    x_guest_session_id: str = Header(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the current usage count for a guest session.
    Allows the frontend to re-sync the counter if localStorage was cleared.
    """
    result = await db.execute(
        select(GuestSession).where(GuestSession.session_id == x_guest_session_id)
    )
    session = result.scalar_one_or_none()
    used = session.generation_count if session else 0
    return {"used": used, "limit": GUEST_GEN_LIMIT}


# ── POST /auth/claim-guest-generations ────────────────────────────────────────

class ClaimRequest(BaseModel):
    guest_session_id: str


@router.post("/auth/claim-guest-generations")
async def claim_guest_generations(
    body: ClaimRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Migrates unexpired guest generations to the authenticated user's account:
      1. Moves each file in R2: guests/{session_id}/… → generations/{user_id}/…
      2. Creates a permanent Generation record in the generations table
      3. Deletes the GuestGeneration record

    Called by the frontend immediately after login / signup.
    Returns { "claimed": N } — gracefully returns 0 if TTL already expired.
    """
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
            continue  # File already gone — skip silently

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
    """Deletes expired, unclaimed guest generations from R2 and the DB."""
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
async def guest_cleanup(db: AsyncSession = Depends(get_db)):
    """
    Deletes expired, unclaimed guest generations.
    No auth — safe to call from an external daily cron.
    """
    deleted = await _run_cleanup(db)
    return {"deleted": deleted}


# ── GET /guest/generations/{id}/image — local mode only ───────────────────────

@router.get("/guest/generations/{generation_id}/image")
async def serve_guest_generation(
    generation_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Serves a guest generation image from local disk (local storage mode only)."""
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
