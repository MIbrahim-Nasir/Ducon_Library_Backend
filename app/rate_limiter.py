"""
Simple in-memory rate limiter for FastAPI endpoints.
No external dependencies — uses a plain dict + timestamps.
Python's GIL protects dict mutations in CPython, so no explicit lock is needed
for the per-request hot path.  The periodic cleanup call (once per minute or so)
is safe as well since it only runs on the event loop thread.
"""
import time
from collections import defaultdict
from fastapi import HTTPException, Request

# { key -> [timestamp, timestamp, ...] }
_attempts: dict[str, list[float]] = defaultdict(list)


def _prune(key: str, window: int, now: float) -> list[float]:
    """Return only timestamps within the sliding window, mutating the dict."""
    cutoff = now - window
    pruned = [t for t in _attempts[key] if t > cutoff]
    _attempts[key] = pruned
    return pruned


def is_rate_limited(key: str, max_requests: int, window_seconds: int) -> bool:
    """
    Returns True if the caller should be blocked, False if the request is allowed.
    Side-effect: records this request timestamp when allowed.
    """
    now = time.time()
    current = _prune(key, window_seconds, now)
    if len(current) >= max_requests:
        return True
    _attempts[key].append(now)
    return False


def cleanup_old_keys(max_age_seconds: int = 3600) -> int:
    """Remove entries not seen in the last max_age_seconds. Returns number of keys removed."""
    now = time.time()
    cutoff = now - max_age_seconds
    stale = [k for k, times in _attempts.items() if not times or max(times) < cutoff]
    for k in stale:
        del _attempts[k]
    return len(stale)


def require_rate_limit(
    request: Request,
    *,
    max_requests: int,
    window_seconds: int,
    key_prefix: str = "",
) -> None:
    """
    Dependency / inline helper that raises 429 when the per-IP limit is exceeded.

    Usage in a route:
        from app.rate_limiter import require_rate_limit
        ...
        client_ip = request.client.host if request.client else "unknown"
        require_rate_limit(request, max_requests=5, window_seconds=60, key_prefix="login")
    """
    client_ip = request.client.host if request.client else "unknown"
    key = f"{key_prefix}:{client_ip}"
    if is_rate_limited(key, max_requests, window_seconds):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait and try again.",
        )
