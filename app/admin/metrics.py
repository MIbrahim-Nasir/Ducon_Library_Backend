"""Aggregation queries for the admin dashboard.

Reads from api_usage_events (raw) and usage_daily_rollup (pre-aggregated, fast)
plus users / generations / session_activity. Heavy queries prefer the rollup
table for ranges > 1 day; recent/intra-day queries hit the raw events table.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.db.models import (
    AdminAuditLog,
    ApiUsageEvent,
    Generation,
    GuestGeneration,
    SessionActivity,
    UsageDailyRollup,
    User,
)

PERCENTILES = [0.5, 0.9]  # we compute median (0.5) in Python where SQL lacks it


def _dt_range(days: int):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return start, end


# ── Overview (totals) ─────────────────────────────────────────────────────────


async def overview(db: AsyncSession) -> dict:
    """Headline totals across all time."""
    total_users = (await db.execute(select(func.count(User.id)))).scalar_one()
    total_generations = (await db.execute(select(func.count(Generation.id)))).scalar_one()
    total_guest_generations = (await db.execute(select(func.count(GuestGeneration.id)))).scalar_one()

    usage_totals = (await db.execute(
        select(
            func.count(ApiUsageEvent.id),
            func.coalesce(func.sum(ApiUsageEvent.input_tokens), 0),
            func.coalesce(func.sum(ApiUsageEvent.output_tokens), 0),
            func.coalesce(func.sum(ApiUsageEvent.image_count), 0),
            func.coalesce(func.sum(ApiUsageEvent.cost_usd), 0),
        )
    )).one()
    total_calls, total_in, total_out, total_imgs, total_cost = usage_totals

    return {
        "total_users": int(total_users or 0),
        "total_generations": int(total_generations or 0),
        "total_guest_generations": int(total_guest_generations or 0),
        "total_ai_calls": int(total_calls or 0),
        "total_input_tokens": int(total_in or 0),
        "total_output_tokens": int(total_out or 0),
        "total_tokens": int((total_in or 0) + (total_out or 0)),
        "total_images_generated": int(total_imgs or 0),
        "total_cost_usd": float(total_cost or 0),
    }


# ── Per-user aggregates ───────────────────────────────────────────────────────


async def per_user_aggregates(db: AsyncSession) -> dict:
    """Avg / median generations, tokens, cost per user.

    Computed in Python from per-user aggregates (small enough for v1; for very
    large user bases we'd push this to the rollup table or a materialized view).
    """
    # generations per user
    rows = (await db.execute(
        select(User.id, func.count(Generation.id)).outerjoin(Generation).group_by(User.id)
    )).all()
    gen_counts = [int(r[1] or 0) for r in rows]

    # tokens / cost per user (from usage events)
    usage_rows = (await db.execute(
        select(
            ApiUsageEvent.user_id,
            func.coalesce(func.sum(ApiUsageEvent.input_tokens), 0),
            func.coalesce(func.sum(ApiUsageEvent.output_tokens), 0),
            func.coalesce(func.sum(ApiUsageEvent.cost_usd), 0),
        ).where(ApiUsageEvent.user_id.isnot(None)).group_by(ApiUsageEvent.user_id)
    )).all()
    tok_counts = [int((r[1] or 0) + (r[2] or 0)) for r in usage_rows]
    cost_vals = [float(r[3] or 0) for r in usage_rows]

    def median(xs):
        if not xs:
            return 0
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else 0

    user_count = len(gen_counts) or 1
    return {
        "generations_per_user": {
            "mean": round(mean(gen_counts), 2),
            "median": median(gen_counts),
            "max": max(gen_counts) if gen_counts else 0,
        },
        "tokens_per_user": {
            "mean": round(mean(tok_counts), 2),
            "median": median(tok_counts),
        },
        "cost_per_user_usd": {
            "mean": round(mean(cost_vals), 6),
            "median": round(median(cost_vals), 6),
            "total": round(sum(cost_vals), 6),
        },
    }


# ── Active users / retention ──────────────────────────────────────────────────


async def active_users(db: AsyncSession, days: int) -> dict:
    """DAU/WAU/MAU + new users in the window."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    active = (await db.execute(
        select(func.count(func.distinct(ApiUsageEvent.user_id))).where(
            ApiUsageEvent.created_at >= start, ApiUsageEvent.user_id.isnot(None)
        )
    )).scalar_one()
    new_users = (await db.execute(
        select(func.count(User.id)).where(User.created_at >= start)
    )).scalar_one()
    return {
        "days": days,
        "active_users": int(active or 0),
        "new_users": int(new_users or 0),
    }


async def daily_active_series(db: AsyncSession, days: int) -> list[dict]:
    """DAU + daily new users + daily cost + daily generations for last N days."""
    start, end = _dt_range(days)
    # DAU from usage events
    dau_rows = (await db.execute(
        select(
            func.date_trunc("day", ApiUsageEvent.created_at).label("d"),
            func.count(func.distinct(ApiUsageEvent.user_id)).label("users"),
            func.count(ApiUsageEvent.id).label("calls"),
            func.coalesce(func.sum(ApiUsageEvent.cost_usd), 0).label("cost"),
            func.coalesce(func.sum(ApiUsageEvent.input_tokens + ApiUsageEvent.output_tokens), 0).label("tokens"),
        ).where(ApiUsageEvent.created_at >= start, ApiUsageEvent.user_id.isnot(None))
        .group_by("d").order_by("d")
    )).all()
    # daily generations
    gen_rows = (await db.execute(
        select(
            func.date_trunc("day", Generation.generated_at).label("d"),
            func.count(Generation.id).label("gens"),
        ).where(Generation.generated_at >= start).group_by("d")
    )).all()
    gen_map = {r.d.date(): int(r.gens) for r in gen_rows}
    out = []
    for r in dau_rows:
        d = r.d.date() if hasattr(r.d, "date") else r.d
        out.append({
            "day": str(d),
            "active_users": int(r.users or 0),
            "ai_calls": int(r.calls or 0),
            "cost_usd": float(r.cost or 0),
            "tokens": int(r.tokens or 0),
            "generations": gen_map.get(d, 0),
        })
    return out


async def retention_cohort(db: AsyncSession, cohort_weeks: int = 8) -> list[dict]:
    """Weekly signup cohort retention based on usage activity.

    cohort = signup week; for each subsequent week, count users from that cohort
    who had any usage event that week. Returns one row per cohort with a
    retention array.
    """
    # Bound cohort_weeks defensively (the route param is also bounded) so a huge
    # value can't force an unbounded scan, and use a real bind parameter for the
    # interval instead of string interpolation.
    cohort_weeks = max(1, min(int(cohort_weeks), 52))
    sql = text("""
        WITH cohorts AS (
            SELECT
                id AS user_id,
                DATE_TRUNC('week', created_at) AS cohort_week
            FROM users
            WHERE created_at >= NOW() - (:weeks * INTERVAL '1 week')
        ),
        active_weeks AS (
            SELECT DISTINCT
                c.user_id,
                c.cohort_week,
                DATE_TRUNC('week', e.created_at) AS active_week
            FROM cohorts c
            LEFT JOIN api_usage_events e
              ON e.user_id = c.user_id
             AND e.created_at >= c.cohort_week
             AND e.created_at <  c.cohort_week + (:weeks * INTERVAL '1 week')
        )
        SELECT
            cohort_week,
            COUNT(DISTINCT user_id) AS cohort_size,
            EXTRACT(WEEK FROM active_week - cohort_week)::int AS week_offset,
            COUNT(DISTINCT user_id) FILTER (WHERE active_week IS NOT NULL) AS active_count
        FROM active_weeks
        GROUP BY cohort_week, week_offset
        ORDER BY cohort_week, week_offset
    """)
    try:
        rows = (await db.execute(sql, {"weeks": cohort_weeks})).all()
    except Exception:
        return []
    cohorts: dict[str, dict] = {}
    for r in rows:
        cw = str(r.cohort_week.date()) if hasattr(r.cohort_week, "date") else str(r.cohort_week)
        c = cohorts.setdefault(cw, {"cohort_week": cw, "cohort_size": int(r.cohort_size or 0), "retention": [0] * cohort_weeks})
        off = int(r.week_offset or 0)
        if 0 <= off < cohort_weeks:
            c["retention"][off] = int(r.active_count or 0)
    return list(cohorts.values())


# ── Per-agent / per-model breakdown ───────────────────────────────────────────


async def agent_breakdown(db: AsyncSession, days: int) -> list[dict]:
    start, _ = _dt_range(days)
    rows = (await db.execute(
        select(
            ApiUsageEvent.agent,
            func.count(ApiUsageEvent.id),
            func.coalesce(func.sum(ApiUsageEvent.input_tokens + ApiUsageEvent.output_tokens), 0),
            func.coalesce(func.sum(ApiUsageEvent.image_count), 0),
            func.coalesce(func.sum(ApiUsageEvent.cost_usd), 0),
        ).where(ApiUsageEvent.created_at >= start)
        .group_by(ApiUsageEvent.agent).order_by(func.sum(ApiUsageEvent.cost_usd).desc())
    )).all()
    return [
        {
            "agent": r[0],
            "calls": int(r[1] or 0),
            "tokens": int(r[2] or 0),
            "images": int(r[3] or 0),
            "cost_usd": float(r[4] or 0),
        }
        for r in rows
    ]


async def model_breakdown(db: AsyncSession, days: int) -> list[dict]:
    start, _ = _dt_range(days)
    rows = (await db.execute(
        select(
            ApiUsageEvent.model,
            ApiUsageEvent.provider,
            func.count(ApiUsageEvent.id),
            func.coalesce(func.sum(ApiUsageEvent.input_tokens + ApiUsageEvent.output_tokens), 0),
            func.coalesce(func.sum(ApiUsageEvent.image_count), 0),
            func.coalesce(func.sum(ApiUsageEvent.cost_usd), 0),
        ).where(ApiUsageEvent.created_at >= start)
        .group_by(ApiUsageEvent.model, ApiUsageEvent.provider).order_by(func.sum(ApiUsageEvent.cost_usd).desc())
    )).all()
    return [
        {
            "model": r[0],
            "provider": r[1],
            "calls": int(r[2] or 0),
            "tokens": int(r[3] or 0),
            "images": int(r[4] or 0),
            "cost_usd": float(r[5] or 0),
        }
        for r in rows
    ]


# ── Time on platform ──────────────────────────────────────────────────────────


async def time_spent(db: AsyncSession, days: int) -> dict:
    """Mean & median daily active time per user (minutes) + a daily series.

    Estimated from session_activity heartbeats: per (user, day), active minutes
    ≈ number of heartbeats (one per ~60s) — but capped and deduped to 60s
    intervals. We approximate as COUNT(DISTINCT DATE_TRUNC('minute', occurred_at)).
    The ``daily`` series aggregates those per-user-day minutes by calendar day so
    the admin UI can chart active minutes over time.
    """
    start, _ = _dt_range(days)
    sql = text("""
        SELECT user_id, day, COUNT(*) AS minutes
        FROM (
            SELECT
                user_id,
                DATE_TRUNC('day', occurred_at) AS day,
                DATE_TRUNC('minute', occurred_at) AS minute
            FROM session_activity
            WHERE occurred_at >= :start AND user_id IS NOT NULL
            GROUP BY user_id, day, minute
        ) t
        GROUP BY user_id, day
    """)
    daily_sql = text("""
        SELECT
            DATE_TRUNC('day', occurred_at) AS day,
            COUNT(DISTINCT user_id) AS users,
            COUNT(*) AS minutes
        FROM (
            SELECT
                user_id,
                DATE_TRUNC('day', occurred_at) AS day,
                DATE_TRUNC('minute', occurred_at) AS minute
            FROM session_activity
            WHERE occurred_at >= :start AND user_id IS NOT NULL
            GROUP BY user_id, day, minute
        ) t
        GROUP BY day
        ORDER BY day
    """)
    try:
        rows = (await db.execute(sql, {"start": start})).all()
    except Exception:
        return {"mean_minutes": 0, "median_minutes": 0, "sample_user_days": 0, "daily": []}
    minutes = [int(r.minutes or 0) for r in rows]
    if not minutes:
        return {"mean_minutes": 0, "median_minutes": 0, "sample_user_days": 0, "daily": []}
    minutes.sort()
    n = len(minutes)
    median = minutes[n // 2] if n % 2 else (minutes[n // 2 - 1] + minutes[n // 2]) / 2

    daily = []
    try:
        drows = (await db.execute(daily_sql, {"start": start})).all()
        for r in drows:
            d = r.day.date() if hasattr(r.day, "date") else r.day
            daily.append({
                "day": str(d),
                "active_minutes": int(r.minutes or 0),
                "users": int(r.users or 0),
            })
    except Exception:
        pass

    return {
        "mean_minutes": round(sum(minutes) / n, 1),
        "median_minutes": median,
        "sample_user_days": n,
        "daily": daily,
    }


# ── User management list / detail ─────────────────────────────────────────────


async def list_users(db: AsyncSession, *, search: str = "", role: Optional[str] = None, limit: int = 50, offset: int = 0) -> dict:
    q = select(User)
    if search:
        like = f"%{search}%"
        q = q.where((User.name.ilike(like)) | (User.email.ilike(like)))
    if role:
        q = q.where(User.role == role)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    rows = (await db.execute(q.order_by(User.created_at.desc()).limit(limit).offset(offset))).scalars().all()
    user_ids = [u.id for u in rows]
    # aggregate usage per user in one query
    usage_map: dict[int, dict] = {}
    if user_ids:
        usage_rows = (await db.execute(
            select(
                ApiUsageEvent.user_id,
                func.count(ApiUsageEvent.id),
                func.coalesce(func.sum(ApiUsageEvent.cost_usd), 0),
                func.max(ApiUsageEvent.created_at),
            ).where(ApiUsageEvent.user_id.in_(user_ids)).group_by(ApiUsageEvent.user_id)
        )).all()
        for r in usage_rows:
            usage_map[r[0]] = {"ai_calls": int(r[1] or 0), "cost_usd": float(r[2] or 0), "last_active": str(r[3]) if r[3] else None}
    gen_map: dict[int, int] = {}
    if user_ids:
        gen_rows = (await db.execute(
            select(Generation.user_id, func.count(Generation.id)).where(Generation.user_id.in_(user_ids)).group_by(Generation.user_id)
        )).all()
        gen_map = {r[0]: int(r[1] or 0) for r in gen_rows}
    items = []
    for u in rows:
        items.append({
            "id": u.id,
            "name": u.name,
            "email": u.email,
            "role": u.role,
            "created_at": str(u.created_at) if u.created_at else None,
            "generations": gen_map.get(u.id, 0),
            "ai_calls": usage_map.get(u.id, {}).get("ai_calls", 0),
            "cost_usd": usage_map.get(u.id, {}).get("cost_usd", 0.0),
            "last_active": usage_map.get(u.id, {}).get("last_active"),
            "user_consent": u.user_consent,
        })
    return {"total": int(total or 0), "items": items, "limit": limit, "offset": offset}


async def user_detail(db: AsyncSession, user_id: int) -> Optional[dict]:
    u = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if u is None:
        return None
    usage = (await db.execute(
        select(
            func.count(ApiUsageEvent.id),
            func.coalesce(func.sum(ApiUsageEvent.input_tokens), 0),
            func.coalesce(func.sum(ApiUsageEvent.output_tokens), 0),
            func.coalesce(func.sum(ApiUsageEvent.cost_usd), 0),
            func.coalesce(func.sum(ApiUsageEvent.image_count), 0),
            func.max(ApiUsageEvent.created_at),
        ).where(ApiUsageEvent.user_id == user_id)
    )).one()
    gens = (await db.execute(select(func.count(Generation.id)).where(Generation.user_id == user_id))).scalar_one()
    # last 30 days daily series
    start, _ = _dt_range(30)
    daily = (await db.execute(
        select(
            func.date_trunc("day", ApiUsageEvent.created_at).label("d"),
            func.count(ApiUsageEvent.id),
            func.coalesce(func.sum(ApiUsageEvent.cost_usd), 0),
        ).where(ApiUsageEvent.user_id == user_id, ApiUsageEvent.created_at >= start).group_by("d").order_by("d")
    )).all()
    return {
        "id": u.id,
        "name": u.name,
        "email": u.email,
        "role": u.role,
        "created_at": str(u.created_at) if u.created_at else None,
        "generations": int(gens or 0),
        "ai_calls": int(usage[0] or 0),
        "input_tokens": int(usage[1] or 0),
        "output_tokens": int(usage[2] or 0),
        "cost_usd": float(usage[3] or 0),
        "images_generated": int(usage[4] or 0),
        "last_active": str(usage[5]) if usage[5] else None,
        "daily": [
            {"day": str(r.d.date()) if hasattr(r.d, "date") else str(r.d), "calls": int(r[1] or 0), "cost_usd": float(r[2] or 0)}
            for r in daily
        ],
    }


# ── Audit log ─────────────────────────────────────────────────────────────────


async def audit_log(db: AsyncSession, *, limit: int = 100, offset: int = 0) -> dict:
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    total = (await db.execute(select(func.count(AdminAuditLog.id)))).scalar_one()
    Admin = aliased(User)
    rows = (await db.execute(
        select(AdminAuditLog, Admin.email)
        .outerjoin(Admin, Admin.id == AdminAuditLog.admin_user_id)
        .order_by(AdminAuditLog.created_at.desc())
        .limit(limit).offset(offset)
    )).all()
    items = []
    for r in rows:
        log = r[0]
        admin_email = r[1]
        items.append({
            "id": log.id,
            "admin_user_id": log.admin_user_id,
            "admin_email": admin_email,
            "action": log.action,
            "target": log.target,
            "details": log.details,
            "ip_address": log.ip_address,
            "created_at": str(log.created_at) if log.created_at else None,
        })
    return {"total": int(total or 0), "items": items, "limit": limit, "offset": offset}
