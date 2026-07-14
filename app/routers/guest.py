"""
app/routers/guest.py
────────────────────
Guest-user endpoints:

  POST /guest/session                   — issue signed HttpOnly guest session cookie
  GET  /guest/usage                     — { generations, chat, voice } used/limit
  GET  /guest/gen-count                 — legacy { used, limit } for generations
  POST /auth/claim-guest-generations    — migrates guest generations to an authenticated user
  POST /guest/cleanup                   — deletes expired guest gens + stale designer job rows (cron)
  GET  /guest/generations/{id}/image    — serves guest generation (local file or R2 redirect/proxy)

Guest session identity:
  - POST /guest/session sets an HttpOnly signed cookie (preferred for browsers).
  - X-Guest-Session-Id header still accepted; header overrides cookie when both sent.
  - Cookie is sent automatically on credentialed fetch (credentials: 'include').

Rate limiting:
  - Cloudflare Turnstile on guest generation, chat, multi-image, and voice (WS query param).
  - Per-guest-session limits via fingerprint-first identity (X-Guest-Fingerprint primary).
  - Optional shared IP / subnet caps (GUEST_IP_TOTAL_LIMIT / GUEST_SUBNET_LIMIT; default 0).

Generation limits count one completed output image per pipeline run — internal
eval/retry rounds do not increment the counter.
"""

import asyncio
import hmac
import os
import uuid
from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import storage
from app.auth import get_current_user
from app.config import ENFORCE_TURNSTILE, IS_PRODUCTION
from app.db.database import get_db
from app.hashing import sha256_hex
from app.rate_limiter import require_rate_limit
from app.signed_urls import verify_guest_generation
from app.db.models import Generation, GuestConsentAudit, GuestGeneration, GuestSession
from app.guest_identity import GuestRequestIdentity, build_guest_request_identity
from app.guest_session_token import require_guest_session_id, set_guest_session_cookie
from app.guest_usage import (
    GuestUsageKind,
    enforce_guest_limit,
    get_guest_session_row,
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

    - Non-production (ENV != production): skip unless TURNSTILE_ENFORCE=true.
    - If TURNSTILE_SECRET_KEY is not set:
        • production → fail closed (verification cannot be trusted).
        • local dev  → skip verification.
    - If TURNSTILE_SECRET_KEY is set in production: token MUST be present and valid.
    """
    if not IS_PRODUCTION and not ENFORCE_TURNSTILE:
        return True
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
    *,
    identity: GuestRequestIdentity,
    usage_kind: GuestUsageKind = GuestUsageKind.GENERATION,
) -> GuestSession:
    """
    Fetch/create guest session and enforce limits for the requested feature.

    ``identity`` bundles client IP, fingerprint, and server-side composite signals.

    Used by generation, chat, and voice entry points.
    """
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid X-Guest-Session-Id. Must be a valid UUID.")
    return await enforce_guest_limit(
        db,
        session_id,
        identity.ip_hash,
        usage_kind,
        fingerprint_hash=identity.fingerprint_hash,
        composite_hash=identity.composite_hash,
        subnet_key=identity.subnet_key,
        ja4_fingerprint=identity.ja4_fingerprint,
        asn=identity.asn,
    )


async def resolve_guest_context(
    request: Request,
    db: AsyncSession,
    *,
    turnstile_token: str | None,
    endpoint: str,
    source: str,
    usage_kind: GuestUsageKind = GuestUsageKind.GENERATION,
) -> GuestSession:
    """Shared guest bootstrap for generation/chat/job entry points:
    session id (header/cookie) → request identity → Turnstile → limit-enforced
    GuestSession row.

    Returns the canonical row — its ``session_id`` may differ from the
    header/cookie after fingerprint remapping, so callers must use
    ``row.session_id`` for all downstream persistence.
    Raises 401/400 on bad session, 403 on Turnstile failure, 429 on limits.
    """
    from app.error_logger import log_warning
    from app.guest_session_token import require_guest_session_id

    guest_session_id = require_guest_session_id(request)
    identity = build_guest_request_identity(
        request.headers,
        peer_host=request.client.host if request.client else None,
        raw_fingerprint=request.headers.get("x-guest-fingerprint"),
    )
    if not await verify_turnstile(turnstile_token or "", identity.client_ip):
        await log_warning(
            "guest",
            f"{source}.verify_turnstile",
            "Turnstile verification failed",
            guest_session_id=guest_session_id,
            endpoint=endpoint,
            http_status=403,
        )
        raise HTTPException(
            status_code=403,
            detail="Bot verification failed. Please try again.",
        )
    return await get_or_create_guest_session(
        db, guest_session_id, identity=identity, usage_kind=usage_kind
    )


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


# ── POST /guest/session ────────────────────────────────────────────────────────

@router.post("/guest/session")
async def create_guest_session(response: Response):
    """
    Issue a server-generated guest session UUID as a signed HttpOnly cookie.

    Frontend integration:
      1. Call POST /guest/session once (credentials: 'include') before guest API calls.
      2. Omit X-Guest-Session-Id — the cookie is sent automatically.
      3. Optionally send X-Guest-Session-Id to override the cookie (e.g. migration).
    """
    session_id = str(uuid.uuid4())
    set_guest_session_cookie(response, session_id)
    return {"session_id": session_id}


# ── GET /guest/usage ───────────────────────────────────────────────────────────

@router.get("/guest/usage")
async def guest_usage(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Returns per-feature usage for a guest session."""
    x_guest_session_id = require_guest_session_id(request)
    identity = build_guest_request_identity(
        request.headers,
        peer_host=request.client.host if request.client else None,
        raw_fingerprint=request.headers.get("x-guest-fingerprint"),
    )
    session = await get_guest_session_row(
        db,
        x_guest_session_id,
        identity.ip_hash,
        fingerprint_hash=identity.fingerprint_hash,
        composite_hash=identity.composite_hash,
        subnet_key=identity.subnet_key,
        ja4_fingerprint=identity.ja4_fingerprint,
        asn=identity.asn,
        create=False,
    )
    return usage_snapshot(session)


# ── GET /guest/gen-count (legacy) ──────────────────────────────────────────────

@router.get("/guest/gen-count")
async def guest_gen_count(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Legacy endpoint — generation counts only."""
    x_guest_session_id = require_guest_session_id(request)
    identity = build_guest_request_identity(
        request.headers,
        peer_host=request.client.host if request.client else None,
        raw_fingerprint=request.headers.get("x-guest-fingerprint"),
    )
    session = await get_guest_session_row(
        db,
        x_guest_session_id,
        identity.ip_hash,
        fingerprint_hash=identity.fingerprint_hash,
        composite_hash=identity.composite_hash,
        subnet_key=identity.subnet_key,
        ja4_fingerprint=identity.ja4_fingerprint,
        asn=identity.asn,
        create=False,
    )
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
    await require_rate_limit(
        request,
        max_requests=20,
        window_seconds=60,
        key_prefix="claim_guest",
        key_suffix=str(current_user.id),
    )

    now = datetime.now(timezone.utc)

    # Prefer fingerprint-canonical session_id when the client still holds a
    # rotated UUID (same device). Do not create a new session during claim.
    claim_session_id = body.guest_session_id
    raw_fp = request.headers.get("x-guest-fingerprint")
    if raw_fp:
        from app.guest_usage import normalise_fingerprint_hash

        fp_hash = normalise_fingerprint_hash(raw_fp)
        if fp_hash:
            fp_result = await db.execute(
                select(GuestSession).where(GuestSession.fingerprint_hash == fp_hash)
            )
            existing = fp_result.scalar_one_or_none()
            if existing is not None:
                claim_session_id = existing.session_id

    result = await db.execute(
        select(GuestGeneration).where(
            GuestGeneration.guest_session_id == claim_session_id,
            GuestGeneration.user_id.is_(None),
            GuestGeneration.expires_at > now,
        )
    )
    guest_generations = result.scalars().all()

    claimed_ids = []
    for gen in guest_generations:
        try:
            new_key = storage.move_guest_to_user(
                session_id=claim_session_id,
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

async def _run_cleanup(db: AsyncSession) -> dict[str, int]:
    from datetime import datetime, timezone

    from app.designer_cleanup import cleanup_designer_jobs

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

    guest_deleted = 0
    if expired:
        await db.execute(
            delete(GuestGeneration).where(
                GuestGeneration.id.in_([g.id for g in expired])
            )
        )
        guest_deleted = len(expired)

    designer_stats = await cleanup_designer_jobs(db)

    # Generation jobs stranded by a worker restart mid-run: without this sweep
    # they stay status='running' forever and cross-worker SSE pollers never
    # see a terminal status. Same policy as designer jobs.
    from sqlalchemy import text as _sql_text
    stale_gen = await db.execute(
        _sql_text(
            "UPDATE generation_jobs "
            "SET status = 'failed', error = 'stale: worker restarted mid-run', "
            "    updated_at = NOW() "
            "WHERE status IN ('queued', 'running') "
            "  AND updated_at < NOW() - INTERVAL '2 hours'"
        )
    )
    generation_jobs_marked_stale = stale_gen.rowcount or 0

    if (
        guest_deleted
        or generation_jobs_marked_stale
        or designer_stats.get("designer_jobs_deleted")
        or designer_stats.get("designer_jobs_marked_stale")
    ):
        await db.commit()

    return {
        "guest_generations_deleted": guest_deleted,
        "generation_jobs_marked_stale": generation_jobs_marked_stale,
        **designer_stats,
    }


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
    return deleted


# ── GET /guest/generations/{id}/image ─────────────────────────────────────────

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

    if storage.CLOUD_STORAGE:
        if storage.should_proxy_generation_images():
            image_bytes = await asyncio.to_thread(
                storage.read_generation_bytes, gen.url
            )
            if not image_bytes:
                raise HTTPException(
                    status_code=404,
                    detail="Generation file not found in cloud storage",
                )
            return Response(
                content=image_bytes,
                media_type="image/png",
                headers=storage.image_response_headers(),
            )
        signed_url = storage.get_guest_generation_url(gen.id, gen.url)
        return RedirectResponse(
            url=signed_url,
            status_code=302,
            headers=storage.image_response_headers(),
        )

    path = storage.serve_local_path(gen.url)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Generation file not found")

    return FileResponse(
        path,
        media_type="image/png",
        headers=storage.image_response_headers(),
    )
