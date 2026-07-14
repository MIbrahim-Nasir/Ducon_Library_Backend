"""Admin / analytics authentication: JWT role + short-lived admin session token.

Roles (``users.role``):
  • ``admin``     — full access: settings, secrets, user CRUD, metrics, errors
  • ``analytics`` — read-only dashboards: metrics, usage, error logs (no settings/secrets/user CRUD)

Flow:
  1. User logs in normally → JWT. Must have role ``admin`` or ``analytics``.
  2. Frontend /admin route prompts for ADMIN_PASSWORD → POST /admin/auth/verify
     → returns a short-lived admin session token (signed JWT with
     admin_session=true claim, TTL ~30min).
  3. Frontend stores admin session token in memory only, sends it as
     X-Admin-Session header on all /admin/* requests alongside the Bearer JWT.
  4. ``require_admin_or_analytics`` validates BOTH: privileged JWT role AND a
     valid X-Admin-Session token. ``require_admin`` is stricter (admin only).

Admin session tokens are revoked in-memory (cleared on restart, like the
existing JWT revocation list).

Env vars:
  ADMIN_PASSWORD_HASH       — bcrypt hash for /admin/auth/verify (required in prod)
  ADMIN_SESSION_TTL_MINUTES — session token lifetime (default 30)
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


# ── Roles ─────────────────────────────────────────────────────────────────────

ROLE_ADMIN = "admin"
ROLE_ANALYTICS = "analytics"
PRIVILEGED_ROLES = frozenset({ROLE_ADMIN, ROLE_ANALYTICS})


def is_admin_role(role: str) -> bool:
    return role == ROLE_ADMIN


def is_analytics_role(role: str) -> bool:
    return role == ROLE_ANALYTICS


def has_privileged_role(role: str) -> bool:
    return role in PRIVILEGED_ROLES


def _validate_admin_session(
    x_admin_session: Optional[str],
    current_user: User,
) -> None:
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
    if str(current_user.id) != str(payload.get("sub")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin session user mismatch",
        )


# ── Dependencies ──────────────────────────────────────────────────────────────


async def require_admin(
    x_admin_session: Optional[str] = Header(default=None, alias="X-Admin-Session"),
    current_user: User = Depends(get_current_user),
) -> User:
    """Require role=admin with a valid admin session token (destructive ops)."""
    if not is_admin_role(current_user.role):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    _validate_admin_session(x_admin_session, current_user)
    return current_user


async def require_admin_or_analytics(
    x_admin_session: Optional[str] = Header(default=None, alias="X-Admin-Session"),
    current_user: User = Depends(get_current_user),
) -> User:
    """Require role=admin or analytics with a valid admin session token."""
    if not has_privileged_role(current_user.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or analytics role required",
        )
    _validate_admin_session(x_admin_session, current_user)
    return current_user


async def require_analytics_jwt_only(current_user: User = Depends(get_current_user)) -> User:
    """Read-only metrics: JWT with role admin or analytics (no admin session password)."""
    if not has_privileged_role(current_user.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or analytics role required",
        )
    return current_user


async def require_admin_user_only(current_user: User = Depends(get_current_user)) -> User:
    """Lighter check: admin or analytics JWT only (no session). Used to bootstrap
    the admin session (e.g. /admin/auth/verify)."""
    if not has_privileged_role(current_user.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or analytics role required",
        )
    return current_user
