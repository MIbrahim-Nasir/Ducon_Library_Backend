"""
Guest session cookie — server-issued, HMAC-signed UUID.

Clients may obtain a session via POST /guest/session (HttpOnly cookie) or
continue sending X-Guest-Session-Id. Header takes precedence when both are present.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from typing import Mapping, Optional

from fastapi import HTTPException, Request, Response

from app.config import IS_PRODUCTION

GUEST_SESSION_COOKIE = "ducon_guest_session"
_TOKEN_SEP = "."

_SIGN_SECRET = (
    os.getenv("GUEST_SESSION_SECRET")
    or os.getenv("JWT_SECRET_KEY", "changeme-use-env-var-in-production")
)


def _sign(session_id: str) -> str:
    return hmac.new(
        _SIGN_SECRET.encode(),
        session_id.encode(),
        hashlib.sha256,
    ).hexdigest()


def sign_guest_session_token(session_id: str) -> str:
    """Return ``{uuid}.{hmac}`` suitable for the guest session cookie value."""
    uuid.UUID(session_id)
    return f"{session_id}{_TOKEN_SEP}{_sign(session_id)}"


def verify_guest_session_token(token: str) -> Optional[str]:
    """Validate cookie value; return the embedded UUID or None if tampered."""
    if not token or _TOKEN_SEP not in token:
        return None
    session_id, sig = token.split(_TOKEN_SEP, 1)
    if not session_id or not sig:
        return None
    try:
        uuid.UUID(session_id)
    except ValueError:
        return None
    if not hmac.compare_digest(_sign(session_id), sig):
        return None
    return session_id


def resolve_guest_session_id(
    request: Request,
    *,
    header_name: str = "x-guest-session-id",
) -> Optional[str]:
    """Header override first, then verified HttpOnly cookie."""
    header = request.headers.get(header_name)
    if header and header.strip():
        return header.strip()
    cookie = request.cookies.get(GUEST_SESSION_COOKIE)
    if cookie:
        return verify_guest_session_token(cookie)
    return None


def resolve_guest_session_id_from_parts(
    headers: Mapping[str, str],
    cookies: Mapping[str, str],
    *,
    header_name: str = "x-guest-session-id",
) -> Optional[str]:
    """WebSocket / non-Request contexts — same precedence as resolve_guest_session_id."""
    header = headers.get(header_name) or headers.get(header_name.lower())
    if header and header.strip():
        return header.strip()
    cookie = cookies.get(GUEST_SESSION_COOKIE)
    if cookie:
        return verify_guest_session_token(cookie)
    return None


def require_guest_session_id(request: Request) -> str:
    """Resolve guest session id or raise 400/401 with stable error messages."""
    raw = resolve_guest_session_id(request)
    if not raw:
        raise HTTPException(
            status_code=401,
            detail=(
                "Authentication required. Provide a Bearer token, "
                "X-Guest-Session-Id header, or guest session cookie."
            ),
        )
    try:
        uuid.UUID(raw)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid X-Guest-Session-Id. Must be a valid UUID.",
        )
    return raw


def set_guest_session_cookie(response: Response, session_id: str) -> None:
    """Attach the signed guest session cookie to a response."""
    response.set_cookie(
        key=GUEST_SESSION_COOKIE,
        value=sign_guest_session_token(session_id),
        httponly=True,
        secure=IS_PRODUCTION,
        samesite="none" if IS_PRODUCTION else "lax",
        max_age=365 * 24 * 3600,
        path="/",
    )
