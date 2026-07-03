"""Admin router: auth, settings, metrics, user management, audit log.

All /admin/* routes (except /admin/auth/verify) require the ``require_admin``
dependency: a valid Bearer JWT with role=admin AND a valid X-Admin-Session
header (short-lived admin session token obtained via /admin/auth/verify).
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin import metrics as M
from app.admin.admin_auth import (
    create_admin_session_token,
    decode_admin_session_token,
    is_admin_password_configured,
    revoke_admin_session,
    verify_admin_password,
    _ADMIN_SESSION_TTL_MIN,
    require_admin,
    require_admin_user_only,
)
from app.admin.pricing import list_pricing
from app.admin.schemas import (
    AdminVerifyRequest,
    AdminVerifyResponse,
    SettingsUpdateRequest,
    SettingsUpdateResponse,
    UserPatchRequest,
)
from app.admin.settings_catalog import (
    EDITABLE_NAMESPACES,
    cast_value,
    get_namespace,
    get_spec,
)
from app.admin.settings_store import get_settings_store
from app.admin.usage_recorder import get_usage_recorder
from app.auth import get_current_user
from app.db.database import get_db
from app.db.models import AdminAuditLog, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ── helpers ───────────────────────────────────────────────────────────────────


def _client_ip(request: Request) -> Optional[str]:
    if request.client:
        return request.client.host
    return None


async def _audit(
    db: AsyncSession,
    admin_user_id: int,
    action: str,
    target: Optional[str] = None,
    details: Optional[dict] = None,
    request: Optional[Request] = None,
) -> None:
    log = AdminAuditLog(
        admin_user_id=admin_user_id,
        action=action,
        target=target,
        details=details,
        ip_address=_client_ip(request) if request else None,
    )
    db.add(log)
    await db.commit()


# ── auth ──────────────────────────────────────────────────────────────────────


@router.post("/auth/verify", response_model=AdminVerifyResponse)
async def admin_verify(
    request: Request,
    payload: AdminVerifyRequest,
    current_user: User = Depends(require_admin_user_only),
    db: AsyncSession = Depends(get_db),
):
    """Verify the admin password and issue a short-lived admin session token.

    Requires the caller to already be authenticated with role=admin (Bearer JWT).
    """
    from app.rate_limiter import require_rate_limit
    require_rate_limit(request, max_requests=5, window_seconds=60, key_prefix="admin_verify")

    if not is_admin_password_configured():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ADMIN_PASSWORD_HASH is not configured on the server.",
        )
    if not verify_admin_password(payload.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin password",
        )
    token = create_admin_session_token(current_user.id)
    await _audit(db, current_user.id, "admin.login", target=current_user.email, request=request)
    return AdminVerifyResponse(admin_session_token=token, expires_in_minutes=_ADMIN_SESSION_TTL_MIN)


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def admin_logout(
    x_admin_session: Optional[str] = Header(default=None, alias="X-Admin-Session"),
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    # require_admin already validated the header; revoke its jti so it can't be
    # reused. Reading the header via Header() (not a bare param) is required —
    # a bare param would be parsed as a query string and always be None here.
    if x_admin_session:
        p = decode_admin_session_token(x_admin_session)
        if p:
            revoke_admin_session(p.get("jti", ""), float(p.get("exp", 0)))
    await _audit(db, current_user.id, "admin.logout", target=current_user.email)


@router.get("/auth/status")
async def admin_status(current_user: User = Depends(require_admin)):
    """Confirm admin session is valid. Returns basic info."""
    return {
        "ok": True,
        "user_id": current_user.id,
        "email": current_user.email,
        "role": current_user.role,
        "password_configured": is_admin_password_configured(),
    }


# ── settings ──────────────────────────────────────────────────────────────────


@router.get("/settings")
async def get_settings(current_user: User = Depends(require_admin)):
    """All settings grouped by namespace. Secrets are masked (value=None)."""
    store = get_settings_store()
    snapshot = store.snapshot()
    by_ns: dict[str, dict] = {}
    for item in snapshot:
        ns = by_ns.setdefault(item["namespace"], {
            "name": item["namespace"],
            "label": item["namespace_label"],
            "settings": [],
        })
        # drop the namespace_label from per-item to avoid repetition
        item2 = {k: v for k, v in item.items() if k != "namespace_label"}
        ns["settings"].append(item2)
    # preserve canonical order
    ordered = [by_ns[ns.name] for ns in [*EDITABLE_NAMESPACES, get_namespace("secrets")] if ns.name in by_ns]
    return {"namespaces": ordered}


@router.put("/settings/{namespace}", response_model=SettingsUpdateResponse)
async def update_settings(
    namespace: str,
    payload: SettingsUpdateRequest,
    request: Request,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Bulk-update settings within one namespace. Validates types + bounds."""
    ns = get_namespace(namespace)
    if ns is None or ns.name == "secrets":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown or read-only namespace")
    if not any(n.name == namespace for n in EDITABLE_NAMESPACES):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Namespace is not editable")

    store = get_settings_store()
    updated: list[str] = []
    errors: list[dict] = []
    for item in payload.settings:
        spec = get_spec(item.key)
        if spec is None:
            errors.append({"key": item.key, "error": "unknown key"})
            continue
        if not spec.editable:
            errors.append({"key": item.key, "error": "not editable"})
            continue
        # verify the key belongs to this namespace
        if not any(s.key == item.key for s in ns.settings):
            errors.append({"key": item.key, "error": "key not in namespace"})
            continue
        try:
            cast_value(spec, item.value)  # validate
            await store.put(namespace, item.key, item.value, current_user.id, db)
            updated.append(item.key)
        except ValueError as e:
            errors.append({"key": item.key, "error": str(e)})
        except PermissionError as e:
            errors.append({"key": item.key, "error": str(e)})

    await _audit(
        db, current_user.id, "settings.update",
        target=namespace,
        details={"updated": updated, "errors": errors},
        request=request,
    )
    return SettingsUpdateResponse(updated=updated, errors=errors)


@router.get("/settings/{key}/reveal")
async def reveal_secret(
    key: str,
    request: Request,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Reveal a secret's raw value. Requires admin session (already enforced by
    require_admin). Always audit-logged. Returns the raw env value."""
    spec = get_spec(key)
    if spec is None or not spec.is_secret:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such secret")
    store = get_settings_store()
    val = store.secret_value(key)
    await _audit(db, current_user.id, "secret.reveal", target=key, request=request)
    return {"key": key, "value": val, "is_set": val is not None and val != ""}


# ── metrics ───────────────────────────────────────────────────────────────────


@router.get("/metrics/overview")
async def metrics_overview(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    return await M.overview(db)


@router.get("/metrics/users")
async def metrics_users(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    return await M.per_user_aggregates(db)


@router.get("/metrics/active")
async def metrics_active(
    days: int = 1,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    days = max(1, min(days, 365))
    return await M.active_users(db, days)


@router.get("/metrics/daily")
async def metrics_daily(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    days = max(1, min(days, 365))
    return {"days": days, "series": await M.daily_active_series(db, days)}


@router.get("/metrics/agents")
async def metrics_agents(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    days = max(1, min(days, 365))
    return {"days": days, "agents": await M.agent_breakdown(db, days)}


@router.get("/metrics/models")
async def metrics_models(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    days = max(1, min(days, 365))
    return {"days": days, "models": await M.model_breakdown(db, days)}


@router.get("/metrics/time-spent")
async def metrics_time_spent(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    days = max(1, min(days, 365))
    return await M.time_spent(db, days)


@router.get("/metrics/retention")
async def metrics_retention(
    weeks: int = 8,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    weeks = max(1, min(weeks, 52))
    return {"weeks": weeks, "cohorts": await M.retention_cohort(db, weeks)}


@router.get("/metrics/recorder")
async def metrics_recorder(
    _: User = Depends(require_admin),
):
    """Internal stats about the usage recorder (queue depth, dropped events)."""
    return get_usage_recorder().stats()


@router.get("/pricing")
async def get_pricing(_: User = Depends(require_admin)):
    """Read-only model pricing reference used for cost calculations."""
    return {"models": list_pricing()}


# ── users ─────────────────────────────────────────────────────────────────────


@router.get("/users")
async def admin_list_users(
    search: str = "",
    role: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    return await M.list_users(db, search=search, role=role, limit=limit, offset=offset)


@router.get("/users/{user_id}")
async def admin_user_detail(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    detail = await M.user_detail(db, user_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="User not found")
    return detail


@router.patch("/users/{user_id}")
async def admin_patch_user(
    user_id: int,
    payload: UserPatchRequest,
    request: Request,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select
    from app.db.models import User as UserModel
    user = (await db.execute(select(UserModel).where(UserModel.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    changes: dict = {}
    if payload.role is not None:
        if payload.role not in ("customer", "admin"):
            raise HTTPException(status_code=400, detail="role must be 'customer' or 'admin'")
        if user.id == current_user.id and payload.role != "admin":
            raise HTTPException(status_code=400, detail="You cannot remove your own admin role")
        if user.role != payload.role:
            user.role = payload.role
            changes["role"] = payload.role
    await db.commit()
    await _audit(
        db, current_user.id, "user.role.change",
        target=user.email,
        details=changes or None,
        request=request,
    )
    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "changes": changes,
    }


# ── audit log ─────────────────────────────────────────────────────────────────


@router.get("/audit-log")
async def admin_audit_log(
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    return await M.audit_log(db, limit=limit, offset=offset)
