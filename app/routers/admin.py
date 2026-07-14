"""Admin router: auth, settings, metrics, user management, audit log, error logs.

Route protection:
  • ``require_admin`` — settings, secrets, user CRUD, audit log, error archive
  • ``require_admin_or_analytics`` — admin session bootstrap (auth verify/logout/status)
  • ``require_analytics_jwt_only`` — metrics, pricing, error log read (Bearer JWT only)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin import metrics as M
from app.admin.admin_auth import (
    ROLE_ADMIN,
    ROLE_ANALYTICS,
    create_admin_session_token,
    decode_admin_session_token,
    is_admin_password_configured,
    is_admin_role,
    revoke_admin_session,
    verify_admin_password,
    _ADMIN_SESSION_TTL_MIN,
    require_admin,
    require_admin_or_analytics,
    require_admin_user_only,
    require_analytics_jwt_only,
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
from app.db.database import get_db
from app.db.models import AdminAuditLog, AppErrorLog, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

_VALID_ROLES = ("customer", ROLE_ADMIN, ROLE_ANALYTICS)


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


def _error_summary(row: AppErrorLog) -> dict:
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "severity": row.severity,
        "category": row.category,
        "source": row.source,
        "user_id": row.user_id,
        "guest_session_id": row.guest_session_id,
        "endpoint": row.endpoint,
        "model": row.model,
        "error_type": row.error_type,
        "message": row.message[:500] if row.message else "",
        "http_status": row.http_status,
        "provider_status": row.provider_status,
        "request_id": row.request_id,
    }


def _error_detail(row: AppErrorLog) -> dict:
    out = _error_summary(row)
    out["message"] = row.message
    out["detail"] = row.detail
    out["archived_at"] = row.archived_at.isoformat() if row.archived_at else None
    return out


# ── auth ──────────────────────────────────────────────────────────────────────


@router.post("/auth/verify", response_model=AdminVerifyResponse)
async def admin_verify(
    request: Request,
    payload: AdminVerifyRequest,
    current_user: User = Depends(require_admin_user_only),
    db: AsyncSession = Depends(get_db),
):
    """Verify the admin password and issue a short-lived admin session token.

    Requires the caller to already be authenticated with role=admin or analytics.
    """
    from app.rate_limiter import require_rate_limit
    await require_rate_limit(request, max_requests=5, window_seconds=60, key_prefix="admin_verify")

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
    current_user: User = Depends(require_admin_or_analytics),
    db: AsyncSession = Depends(get_db),
):
    if x_admin_session:
        p = decode_admin_session_token(x_admin_session)
        if p:
            revoke_admin_session(p.get("jti", ""), float(p.get("exp", 0)))
    await _audit(db, current_user.id, "admin.logout", target=current_user.email)


@router.get("/auth/status")
async def admin_status(current_user: User = Depends(require_admin_or_analytics)):
    """Confirm admin/analytics session is valid. Returns role and capabilities."""
    return {
        "ok": True,
        "user_id": current_user.id,
        "email": current_user.email,
        "role": current_user.role,
        "password_configured": is_admin_password_configured(),
        "capabilities": {
            "settings": is_admin_role(current_user.role),
            "secrets": is_admin_role(current_user.role),
            "users": is_admin_role(current_user.role),
            "audit_log": is_admin_role(current_user.role),
            "metrics": True,
            "errors": True,
        },
    }


# ── settings (admin only) ─────────────────────────────────────────────────────


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
        item2 = {k: v for k, v in item.items() if k != "namespace_label"}
        ns["settings"].append(item2)
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
        if not any(s.key == item.key for s in ns.settings):
            errors.append({"key": item.key, "error": "key not in namespace"})
            continue
        try:
            cast_value(spec, item.value)
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
    """Reveal a secret's raw value. Admin only. Always audit-logged."""
    spec = get_spec(key)
    if spec is None or not spec.is_secret:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such secret")
    store = get_settings_store()
    val = store.secret_value(key)
    await _audit(db, current_user.id, "secret.reveal", target=key, request=request)
    return {"key": key, "value": val, "is_set": val is not None and val != ""}


# ── metrics (admin + analytics) ───────────────────────────────────────────────


@router.get("/metrics/overview")
async def metrics_overview(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_analytics_jwt_only),
):
    return await M.overview(db)


@router.get("/metrics/users")
async def metrics_users(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_analytics_jwt_only),
):
    return await M.per_user_aggregates(db)


@router.get("/metrics/active")
async def metrics_active(
    days: int = 1,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_analytics_jwt_only),
):
    days = max(1, min(days, 365))
    return await M.active_users(db, days)


@router.get("/metrics/daily")
async def metrics_daily(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_analytics_jwt_only),
):
    days = max(1, min(days, 365))
    return {"days": days, "series": await M.daily_active_series(db, days)}


@router.get("/metrics/agents")
async def metrics_agents(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_analytics_jwt_only),
):
    days = max(1, min(days, 365))
    return {"days": days, "agents": await M.agent_breakdown(db, days)}


@router.get("/metrics/models")
async def metrics_models(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_analytics_jwt_only),
):
    days = max(1, min(days, 365))
    return {"days": days, "models": await M.model_breakdown(db, days)}


@router.get("/metrics/time-spent")
async def metrics_time_spent(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_analytics_jwt_only),
):
    days = max(1, min(days, 365))
    return await M.time_spent(db, days)


@router.get("/metrics/retention")
async def metrics_retention(
    weeks: int = 8,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_analytics_jwt_only),
):
    weeks = max(1, min(weeks, 52))
    return {"weeks": weeks, "cohorts": await M.retention_cohort(db, weeks)}


@router.get("/metrics/recorder")
async def metrics_recorder(
    _: User = Depends(require_analytics_jwt_only),
):
    """Internal stats about the usage recorder (queue depth, dropped events)."""
    return get_usage_recorder().stats()


@router.get("/pricing")
async def get_pricing(_: User = Depends(require_analytics_jwt_only)):
    """Read-only model pricing reference used for cost calculations."""
    return {"models": list_pricing()}


# ── users (admin only) ────────────────────────────────────────────────────────


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
    from app.db.models import User as UserModel
    user = (await db.execute(select(UserModel).where(UserModel.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    changes: dict = {}
    if payload.role is not None:
        if payload.role not in _VALID_ROLES:
            raise HTTPException(
                status_code=400,
                detail=f"role must be one of: {', '.join(_VALID_ROLES)}",
            )
        if user.id == current_user.id and payload.role not in (ROLE_ADMIN, ROLE_ANALYTICS):
            raise HTTPException(status_code=400, detail="You cannot remove your own privileged role")
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


# ── audit log (admin only) ────────────────────────────────────────────────────


@router.get("/audit-log")
async def admin_audit_log(
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    return await M.audit_log(db, limit=limit, offset=offset)


# ── error logs (read: admin + analytics; archive: admin only) ─────────────────


@router.get("/errors")
async def list_errors(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_analytics_jwt_only),
    category: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    user_id: Optional[int] = Query(None),
    guest_session_id: Optional[str] = Query(None),
    since: Optional[str] = Query(None, description="ISO8601 datetime lower bound"),
    until: Optional[str] = Query(None, description="ISO8601 datetime upper bound"),
    include_archived: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    q = select(AppErrorLog)
    count_q = select(func.count()).select_from(AppErrorLog)

    if not include_archived:
        q = q.where(AppErrorLog.archived_at.is_(None))
        count_q = count_q.where(AppErrorLog.archived_at.is_(None))
    if category:
        q = q.where(AppErrorLog.category == category)
        count_q = count_q.where(AppErrorLog.category == category)
    if severity:
        q = q.where(AppErrorLog.severity == severity)
        count_q = count_q.where(AppErrorLog.severity == severity)
    if user_id is not None:
        q = q.where(AppErrorLog.user_id == user_id)
        count_q = count_q.where(AppErrorLog.user_id == user_id)
    if guest_session_id:
        q = q.where(AppErrorLog.guest_session_id == guest_session_id)
        count_q = count_q.where(AppErrorLog.guest_session_id == guest_session_id)
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            q = q.where(AppErrorLog.created_at >= since_dt)
            count_q = count_q.where(AppErrorLog.created_at >= since_dt)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid since datetime")
    if until:
        try:
            until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
            q = q.where(AppErrorLog.created_at <= until_dt)
            count_q = count_q.where(AppErrorLog.created_at <= until_dt)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid until datetime")

    total = int((await db.execute(count_q)).scalar() or 0)
    rows = (
        await db.execute(
            q.order_by(AppErrorLog.created_at.desc()).limit(limit).offset(offset)
        )
    ).scalars().all()
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [_error_summary(r) for r in rows],
    }


@router.get("/errors/{error_id}")
async def get_error(
    error_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_analytics_jwt_only),
):
    row = (
        await db.execute(select(AppErrorLog).where(AppErrorLog.id == error_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Error log not found")
    return _error_detail(row)


@router.delete("/errors/{error_id}", status_code=status.HTTP_204_NO_CONTENT)
async def archive_error(
    error_id: int,
    request: Request,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Soft-archive an error log entry (admin only)."""
    row = (
        await db.execute(select(AppErrorLog).where(AppErrorLog.id == error_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Error log not found")
    if row.archived_at is None:
        row.archived_at = datetime.now(timezone.utc)
        await db.commit()
        await _audit(
            db, current_user.id, "error.archive",
            target=str(error_id),
            request=request,
        )
