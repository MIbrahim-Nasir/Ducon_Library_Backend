"""Admin authentication: JWT role=admin + short-lived admin session token.

Flow:
  1. User logs in normally → JWT (existing flow). Must have users.role='admin'.
  2. Frontend /admin route prompts for ADMIN_PASSWORD → POST /admin/auth/verify
     → returns a short-lived admin session token (signed JWT with
     admin_session=true claim, TTL ~30min).
  3. Frontend stores admin session token in memory only, sends it as
     X-Admin-Session header on all /admin/* requests alongside the Bearer JWT.
  4. ``require_admin`` dependency validates BOTH: the Bearer JWT user has
     role='admin' AND the X-Admin-Session token is valid and unexpired.

Admin session tokens are revoked in-memory (cleared on restart, like the
existing JWT revocation list).
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import (
    ALGORITHM,
    SECRET_KEY,
    get_current_user,
    pwd_context,
)
from app.config import IS_PRODUCTION
from app.db.database import get_db
from app.db.models import User

# ── Config ────────────────────────────────────────────────────────────────────

_ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")
_ADMIN_SESSION_TTL_MIN = int(os.getenv("ADMIN_SESSION_TTL_MINUTES", "30"))
_ADMIN_ISSUER = "ducon-admin"

# In-memory admin session revocation (jti -> exp). Cleared on restart.
_revoked_admin_jtis: dict[str, float] = {}


def _is_admin_revoked(jti: str) -> bool:
    return jti in _revoked_admin_jtis


def revoke_admin_session(jti: str, exp: float) -> None:
    _revoked_admin_jtis[jti] = exp
    now = datetime.now(timezone.utc).timestamp()
    for k in [k for k, v in _revoked_admin_jtis.items() if v < now]:
        del _revoked_admin_jtis[k]


# ── Admin session tokens ──────────────────────────────────────────────────────


def create_admin_session_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=_ADMIN_SESSION_TTL_MIN)
    return jwt.encode(
        {
            "sub": str(user_id),
            "iss": _ADMIN_ISSUER,
            "admin_session": True,
            "exp": expire,
            "jti": str(uuid.uuid4()),
        },
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def verify_admin_password(plain: str) -> bool:
    if not _ADMIN_PASSWORD_HASH:
        # No admin password configured
        return False
    try:
        return pwd_context.verify(plain, _ADMIN_PASSWORD_HASH)
    except Exception:
        return False


def is_admin_password_configured() -> bool:
    return bool(_ADMIN_PASSWORD_HASH)


def decode_admin_session_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if not payload.get("admin_session"):
            return None
        if payload.get("iss") != _ADMIN_ISSUER:
            return None
        jti = payload.get("jti", "")
        if jti and _is_admin_revoked(jti):
            return None
        return payload
    except JWTError:
        return None


# ── Dependency ────────────────────────────────────────────────────────────────


async def require_admin(
    x_admin_session: Optional[str] = Header(default=None, alias="X-Admin-Session"),
    current_user: User = Depends(get_current_user),
) -> User:
    """Require an authenticated admin user with a valid admin session token.

    Returns the User row. Raises 403 if not admin, 401 if admin session missing
    or invalid.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")

    if not x_admin_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin session token required",
        )
    payload = decode_admin_session_token(x_admin_session)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired admin session",
        )
    # token subject must match the authenticated user
    if str(current_user.id) != str(payload.get("sub")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin session user mismatch",
        )
    return current_user


async def require_admin_user_only(current_user: User = Depends(get_current_user)) -> User:
    """Lighter check: just role=admin (no admin session). Used for endpoints
    that bootstrap the admin session (e.g. /admin/auth/verify)."""
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return current_user
