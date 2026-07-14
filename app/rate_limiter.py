"""
Rate limiter for FastAPI endpoints — Postgres-backed, multi-worker safe.

Counters live in the ``rate_limit_counters`` table (fixed-window, atomic
UPSERT increment), so limits hold exactly across gunicorn workers and survive
restarts — same pattern as ``chat_sessions`` / ``revoked_jtis``.

If the database is unreachable the limiter falls back to the previous
in-memory sliding window (per-worker, best-effort) instead of failing the
request, and logs a warning.
"""
import logging
import time
from collections import defaultdict

from fastapi import HTTPException, Request
from sqlalchemy import text

from app.guest_identity import extract_client_ip

logger = logging.getLogger(__name__)

# ── In-memory fallback (per-worker) ───────────────────────────────────────────
# { key -> [timestamp, timestamp, ...] }
_attempts: dict[str, list[float]] = defaultdict(list)


def _prune(key: str, window: int, now: float) -> list[float]:
    """Return only timestamps within the sliding window, mutating the dict."""
    cutoff = now - window
    pruned = [t for t in _attempts[key] if t > cutoff]
    _attempts[key] = pruned
    return pruned


def is_rate_limited(key: str, max_requests: int, window_seconds: int) -> bool:
    """In-memory fallback check. Records the request timestamp when allowed."""
    now = time.time()
    current = _prune(key, window_seconds, now)
    if len(current) >= max_requests:
        return True
    _attempts[key].append(now)
    return False


def cleanup_old_keys(max_age_seconds: int = 3600) -> int:
    """Remove in-memory entries not seen recently. Returns number removed."""
    now = time.time()
    cutoff = now - max_age_seconds
    stale = [k for k, times in _attempts.items() if not times or max(times) < cutoff]
    for k in stale:
        del _attempts[k]
    return len(stale)


# ── Postgres fixed-window counter ─────────────────────────────────────────────

_UPSERT_SQL = text(
    "INSERT INTO rate_limit_counters AS r (key, window_start, count) "
    "VALUES (:key, :ws, 1) "
    "ON CONFLICT (key) DO UPDATE SET "
    "  count = CASE WHEN r.window_start = :ws THEN r.count + 1 ELSE 1 END, "
    "  window_start = :ws "
    "RETURNING count"
)


async def _db_hit_count(key: str, window_seconds: int) -> int | None:
    """Increment + read the fixed-window counter. None when the DB failed."""
    from app.db.database import async_session_maker

    ws = float(int(time.time() // window_seconds) * window_seconds)
    try:
        async with async_session_maker() as db:
            result = await db.execute(_UPSERT_SQL, {"key": key, "ws": ws})
            count = result.scalar_one()
            await db.commit()
            return int(count)
    except Exception:
        logger.warning("[RateLimit] DB counter failed for %s — in-memory fallback", key)
        return None


async def cleanup_expired_windows(max_age_seconds: int = 3600) -> None:
    """Purge counter rows whose window ended long ago (called from the
    periodic cleanup loop in main.lifespan alongside cleanup_old_keys)."""
    from app.db.database import async_session_maker

    try:
        async with async_session_maker() as db:
            await db.execute(
                text("DELETE FROM rate_limit_counters WHERE window_start < :cutoff"),
                {"cutoff": time.time() - max_age_seconds},
            )
            await db.commit()
    except Exception:
        logger.warning("[RateLimit] cleanup_expired_windows failed", exc_info=True)


def client_ip_for_rate_limit(request: Request) -> str:
    """Prefer Cloudflare / proxy client IP over the immediate peer (often the CDN)."""
    peer = request.client.host if request.client else None
    return extract_client_ip(request.headers, peer_host=peer)


async def require_rate_limit(
    request: Request,
    *,
    max_requests: int,
    window_seconds: int,
    key_prefix: str = "",
    key_suffix: str = "",
) -> None:
    """
    Raises 429 when the limit is exceeded. Postgres-backed (exact across
    workers); in-memory per-worker fallback when the DB is unavailable.

    Keys are ``{key_prefix}:{client_ip}`` or ``{key_prefix}:{client_ip}:{key_suffix}``
    when ``key_suffix`` is set (e.g. email, authenticated user id, or guest session).

    Prefer ``key_suffix`` with a per-user / per-guest id on shared-NAT offices so
    coworkers do not share the same bucket. IP remains in the key as an abuse
    backstop for anonymous traffic without a session.

    Usage in a route:
        from app.rate_limiter import require_rate_limit
        ...
        await require_rate_limit(request, max_requests=5, window_seconds=60, key_prefix="login")
    """
    client_ip = client_ip_for_rate_limit(request)
    suffix = key_suffix or _default_identity_suffix(request)
    key = f"{key_prefix}:{client_ip}"
    if suffix:
        key = f"{key}:{suffix}"

    count = await _db_hit_count(key[:255], window_seconds)
    limited = (count > max_requests) if count is not None else is_rate_limited(
        key, max_requests, window_seconds
    )
    if limited:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait and try again.",
        )


def _default_identity_suffix(request: Request) -> str:
    """Prefer guest session id, else authenticated user id from JWT ``sub``."""
    try:
        from app.guest_session_token import resolve_guest_session_id
        sid = resolve_guest_session_id(request)
        if sid:
            return f"g:{sid}"
    except Exception:
        pass
    auth = request.headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        return ""
    token = auth.split(" ", 1)[1].strip()
    if not token:
        return ""
    try:
        # Signature-verified decode (same secret as auth). Use ``sub`` so each
        # logged-in user gets an isolated bucket — NOT the JWT header bytes,
        # which are identical for every HS256 token.
        from app.auth import ALGORITHM, SECRET_KEY, _is_revoked
        from jose import jwt

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        jti = payload.get("jti") or ""
        if jti and _is_revoked(jti):
            return ""
        sub = payload.get("sub")
        if sub is not None:
            return f"u:{sub}"
    except Exception:
        pass
    return ""
