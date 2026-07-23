"""Aggregation queries for the admin dashboard.

Reads from api_usage_events (raw) plus users / guest_sessions / generations /
session_activity. All day buckets and range windows use **UTC** calendar days.

Date params (shared by metrics endpoints):
  - ``range``: ``7d`` | ``30d`` | ``90d`` | ``ytd`` | ``all`` (default ``30d``)
  - ``from`` / ``to``: inclusive ``YYYY-MM-DD`` (UTC); override presets when both set
  - ``days``: legacy rolling-day count (still supported; treated as calendar days
    ending today inclusive)

Series always include every calendar day in the window with zeros for gaps so
charts do not stop early when recent days have no activity.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.db.models import (
    AdminAuditLog,
    ApiUsageEvent,
    Generation,
    GuestGeneration,
    GuestSession,
    SessionActivity,
    User,
)

# Hard cap on series length (custom / all). ~3 years of daily points.
_MAX_SERIES_DAYS = 1095

_RANGE_ALIASES = {
    "7": "7d",
    "7d": "7d",
    "30": "30d",
    "30d": "30d",
    "90": "90d",
    "90d": "90d",
    "ytd": "ytd",
    "all": "all",
}


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _as_utc_day(value: Any) -> date:
    if value is None:
        raise ValueError("expected date")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).date()
    if isinstance(value, date):
        return value
    if hasattr(value, "date") and callable(value.date):
        return value.date()
    return date.fromisoformat(str(value)[:10])


def _day_start(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _day_end_exclusive(d: date) -> datetime:
    return _day_start(d + timedelta(days=1))


def parse_ymd(value: str) -> date:
    return date.fromisoformat(str(value).strip()[:10])


def resolve_date_window(
    *,
    days: Optional[int] = None,
    range_key: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    now: Optional[datetime] = None,
    earliest_day: Optional[date] = None,
) -> dict:
    """Resolve an inclusive UTC calendar-day window.

    Returns keys: start, end_exclusive, start_day, end_day, day_count, range, timezone.
    ``start`` / ``end_exclusive`` are timezone-aware UTC datetimes suitable for
    ``created_at >= start AND created_at < end_exclusive``.
    """
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(timezone.utc).date()

    start_day: Optional[date] = None
    end_day: Optional[date] = None
    label = "30d"

    if date_from or date_to:
        end_day = parse_ymd(date_to) if date_to else today
        start_day = parse_ymd(date_from) if date_from else end_day
        label = "custom"
    elif range_key:
        key = _RANGE_ALIASES.get(str(range_key).strip().lower())
        if key is None:
            raise ValueError(f"invalid range '{range_key}'")
        label = key
        end_day = today
        if key == "7d":
            start_day = today - timedelta(days=6)
        elif key == "30d":
            start_day = today - timedelta(days=29)
        elif key == "90d":
            start_day = today - timedelta(days=89)
        elif key == "ytd":
            start_day = date(today.year, 1, 1)
        elif key == "all":
            start_day = earliest_day or date(today.year - 3, 1, 1)
            if start_day > today:
                start_day = today
    elif days is not None:
        n = max(1, min(int(days), _MAX_SERIES_DAYS))
        end_day = today
        start_day = today - timedelta(days=n - 1)
        label = f"{n}d"
    else:
        end_day = today
        start_day = today - timedelta(days=29)
        label = "30d"

    assert start_day is not None and end_day is not None
    if end_day > today:
        end_day = today
    if start_day > end_day:
        start_day = end_day

    day_count = (end_day - start_day).days + 1
    if day_count > _MAX_SERIES_DAYS:
        start_day = end_day - timedelta(days=_MAX_SERIES_DAYS - 1)
        day_count = _MAX_SERIES_DAYS
        if label == "custom":
            label = "custom_clamped"

    return {
        "start": _day_start(start_day),
        "end_exclusive": _day_end_exclusive(end_day),
        "start_day": start_day,
        "end_day": end_day,
        "day_count": day_count,
        "range": label,
        "timezone": "UTC",
    }


def iter_days(start_day: date, end_day: date):
    d = start_day
    while d <= end_day:
        yield d
        d += timedelta(days=1)


def fill_daily_series(
    start_day: date,
    end_day: date,
    by_day: dict[date, dict],
    *,
    defaults: Optional[dict] = None,
) -> list[dict]:
    """Build a continuous day series with zeros for missing days."""
    base = dict(defaults or {})
    out: list[dict] = []
    for d in iter_days(start_day, end_day):
        row = {"day": str(d), **base}
        if d in by_day:
            row.update(by_day[d])
        out.append(row)
    return out


def aggregate_series_by_week(series: list[dict], sum_keys: list[str]) -> list[dict]:
    """Roll daily points into ISO weeks (Monday-start), summing ``sum_keys``."""
    buckets: dict[date, dict] = {}
    order: list[date] = []
    for row in series:
        d = parse_ymd(row["day"])
        week_start = d - timedelta(days=d.weekday())
        if week_start not in buckets:
            buckets[week_start] = {
                "day": str(week_start),
                "week_start": str(week_start),
                **{k: 0 for k in sum_keys},
            }
            order.append(week_start)
        bucket = buckets[week_start]
        for k in sum_keys:
            bucket[k] = bucket.get(k, 0) + (row.get(k) or 0)
    return [buckets[w] for w in order]


def _series_averages(series: list[dict], keys: list[str]) -> dict[str, float]:
    n = len(series) or 1
    out: dict[str, float] = {}
    for k in keys:
        total = sum(float(r.get(k) or 0) for r in series)
        out[f"avg_{k}_per_day"] = round(total / n, 4)
        out[f"total_{k}"] = round(total, 4) if isinstance(total, float) else total
    return out


def _window_payload(window: dict) -> dict:
    return {
        "from": str(window["start_day"]),
        "to": str(window["end_day"]),
        "days": window["day_count"],
        "range": window["range"],
        "timezone": window["timezone"],
    }


async def _earliest_activity_day(db: AsyncSession) -> Optional[date]:
    """Earliest day across users, guest sessions, and usage — for ``range=all``."""
    candidates: list[date] = []
    for model, col in (
        (User, User.created_at),
        (GuestSession, GuestSession.created_at),
        (ApiUsageEvent, ApiUsageEvent.created_at),
    ):
        try:
            val = (await db.execute(select(func.min(col)))).scalar_one_or_none()
        except Exception:
            continue
        if val is not None:
            candidates.append(_as_utc_day(val))
    return min(candidates) if candidates else None


async def resolve_metrics_window(
    db: AsyncSession,
    *,
    days: Optional[int] = None,
    range_key: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    earliest = None
    key = (range_key or "").strip().lower() if range_key else None
    if key in ("all",) or _RANGE_ALIASES.get(key or "") == "all":
        earliest = await _earliest_activity_day(db)
    return resolve_date_window(
        days=days,
        range_key=range_key,
        date_from=date_from,
        date_to=date_to,
        earliest_day=earliest,
    )


# ── Overview (totals) ─────────────────────────────────────────────────────────


async def overview(db: AsyncSession) -> dict:
    """Headline totals across all time."""
    total_users = (await db.execute(select(func.count(User.id)))).scalar_one()
    total_generations = (await db.execute(select(func.count(Generation.id)))).scalar_one()
    total_guest_generations = (await db.execute(select(func.count(GuestGeneration.id)))).scalar_one()
    total_guest_sessions = (await db.execute(select(func.count(GuestSession.id)))).scalar_one()

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
        "total_guest_sessions": int(total_guest_sessions or 0),
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
    rows = (await db.execute(
        select(User.id, func.count(Generation.id)).outerjoin(Generation).group_by(User.id)
    )).all()
    gen_counts = [int(r[1] or 0) for r in rows]

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


async def active_users(
    db: AsyncSession,
    *,
    days: Optional[int] = None,
    range_key: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    """Window totals for active/new registered users and guests.

    Definitions (UTC):
      - active_users: distinct ``user_id`` with ≥1 api_usage_events row in window
      - new_users: users with ``created_at`` in window
      - active_guests: distinct ``guest_session_id`` with ≥1 usage event in window
      - new_guests: guest_sessions with ``created_at`` in window
    """
    window = await resolve_metrics_window(
        db, days=days, range_key=range_key, date_from=date_from, date_to=date_to
    )
    start, end = window["start"], window["end_exclusive"]
    day_count = window["day_count"] or 1

    active = (await db.execute(
        select(func.count(func.distinct(ApiUsageEvent.user_id))).where(
            ApiUsageEvent.created_at >= start,
            ApiUsageEvent.created_at < end,
            ApiUsageEvent.user_id.isnot(None),
        )
    )).scalar_one()
    new_users = (await db.execute(
        select(func.count(User.id)).where(
            User.created_at >= start,
            User.created_at < end,
        )
    )).scalar_one()
    active_guests = (await db.execute(
        select(func.count(func.distinct(ApiUsageEvent.guest_session_id))).where(
            ApiUsageEvent.created_at >= start,
            ApiUsageEvent.created_at < end,
            ApiUsageEvent.guest_session_id.isnot(None),
        )
    )).scalar_one()
    new_guests = (await db.execute(
        select(func.count(GuestSession.id)).where(
            GuestSession.created_at >= start,
            GuestSession.created_at < end,
        )
    )).scalar_one()

    nu = int(new_users or 0)
    ng = int(new_guests or 0)
    au = int(active or 0)
    ag = int(active_guests or 0)
    return {
        **_window_payload(window),
        "active_users": au,
        "new_users": nu,
        "active_guests": ag,
        "new_guests": ng,
        "avg_new_users_per_day": round(nu / day_count, 4),
        "avg_new_guests_per_day": round(ng / day_count, 4),
        "avg_active_users_per_day": round(au / day_count, 4),
        "avg_active_guests_per_day": round(ag / day_count, 4),
    }


_DAILY_DEFAULTS = {
    "active_users": 0,
    "active_guests": 0,
    "new_users": 0,
    "new_guests": 0,
    "ai_calls": 0,
    "cost_usd": 0.0,
    "tokens": 0,
    "generations": 0,
    "guest_generations": 0,
}

_DAILY_AVG_KEYS = [
    "active_users",
    "active_guests",
    "new_users",
    "new_guests",
    "ai_calls",
    "cost_usd",
    "tokens",
    "generations",
    "guest_generations",
]


async def daily_active_series(
    db: AsyncSession,
    *,
    days: Optional[int] = None,
    range_key: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    granularity: str = "day",
) -> dict:
    """DAU / guests / cost / generations for a continuous UTC day (or week) series."""
    window = await resolve_metrics_window(
        db, days=days, range_key=range_key, date_from=date_from, date_to=date_to
    )
    start, end = window["start"], window["end_exclusive"]
    start_day, end_day = window["start_day"], window["end_day"]

    by_day: dict[date, dict] = {}

    def _bucket(dval) -> dict:
        d = _as_utc_day(dval)
        if d not in by_day:
            by_day[d] = {}
        return by_day[d]

    # Usage: auth DAU + guest activity + cost/calls (all events)
    # COUNT(DISTINCT user_id/guest_session_id) ignores NULLs in SQL.
    usage_rows = (await db.execute(
        select(
            func.date_trunc("day", ApiUsageEvent.created_at).label("d"),
            func.count(func.distinct(ApiUsageEvent.user_id)).label("users"),
            func.count(func.distinct(ApiUsageEvent.guest_session_id)).label("guests"),
            func.count(ApiUsageEvent.id).label("calls"),
            func.coalesce(func.sum(ApiUsageEvent.cost_usd), 0).label("cost"),
            func.coalesce(
                func.sum(ApiUsageEvent.input_tokens + ApiUsageEvent.output_tokens), 0
            ).label("tokens"),
        )
        .where(ApiUsageEvent.created_at >= start, ApiUsageEvent.created_at < end)
        .group_by("d")
        .order_by("d")
    )).all()
    for r in usage_rows:
        b = _bucket(r.d)
        b["active_users"] = int(r.users or 0)
        b["active_guests"] = int(r.guests or 0)
        b["ai_calls"] = int(r.calls or 0)
        b["cost_usd"] = float(r.cost or 0)
        b["tokens"] = int(r.tokens or 0)

    # New registered users
    new_user_rows = (await db.execute(
        select(
            func.date_trunc("day", User.created_at).label("d"),
            func.count(User.id).label("n"),
        )
        .where(User.created_at >= start, User.created_at < end)
        .group_by("d")
    )).all()
    for r in new_user_rows:
        _bucket(r.d)["new_users"] = int(r.n or 0)

    # New guest sessions
    new_guest_rows = (await db.execute(
        select(
            func.date_trunc("day", GuestSession.created_at).label("d"),
            func.count(GuestSession.id).label("n"),
        )
        .where(GuestSession.created_at >= start, GuestSession.created_at < end)
        .group_by("d")
    )).all()
    for r in new_guest_rows:
        _bucket(r.d)["new_guests"] = int(r.n or 0)

    # Auth generations
    gen_rows = (await db.execute(
        select(
            func.date_trunc("day", Generation.generated_at).label("d"),
            func.count(Generation.id).label("gens"),
        )
        .where(Generation.generated_at >= start, Generation.generated_at < end)
        .group_by("d")
    )).all()
    for r in gen_rows:
        _bucket(r.d)["generations"] = int(r.gens or 0)

    # Guest generations
    ggen_rows = (await db.execute(
        select(
            func.date_trunc("day", GuestGeneration.generated_at).label("d"),
            func.count(GuestGeneration.id).label("gens"),
        )
        .where(
            GuestGeneration.generated_at >= start,
            GuestGeneration.generated_at < end,
        )
        .group_by("d")
    )).all()
    for r in ggen_rows:
        _bucket(r.d)["guest_generations"] = int(r.gens or 0)

    series = fill_daily_series(start_day, end_day, by_day, defaults=_DAILY_DEFAULTS)
    gran = (granularity or "day").strip().lower()
    if gran == "week":
        series = aggregate_series_by_week(series, list(_DAILY_DEFAULTS.keys()))

    averages = _series_averages(series if gran == "day" else fill_daily_series(
        start_day, end_day, by_day, defaults=_DAILY_DEFAULTS
    ), _DAILY_AVG_KEYS)

    return {
        **_window_payload(window),
        "granularity": gran if gran in ("day", "week") else "day",
        "averages": {
            "active_users_per_day": averages["avg_active_users_per_day"],
            "active_guests_per_day": averages["avg_active_guests_per_day"],
            "new_users_per_day": averages["avg_new_users_per_day"],
            "new_guests_per_day": averages["avg_new_guests_per_day"],
            "ai_calls_per_day": averages["avg_ai_calls_per_day"],
            "cost_usd_per_day": averages["avg_cost_usd_per_day"],
            "generations_per_day": averages["avg_generations_per_day"],
            "guest_generations_per_day": averages["avg_guest_generations_per_day"],
        },
        "series": series,
    }


async def retention_cohort(db: AsyncSession, cohort_weeks: int = 8) -> list[dict]:
    """Weekly signup cohort retention based on usage activity.

    cohort = signup week; for each subsequent week, count users from that cohort
    who had any usage event that week. Returns one row per cohort with a
    retention array.
    """
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


async def agent_breakdown(
    db: AsyncSession,
    *,
    days: Optional[int] = None,
    range_key: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    window = await resolve_metrics_window(
        db, days=days, range_key=range_key, date_from=date_from, date_to=date_to
    )
    start, end = window["start"], window["end_exclusive"]
    rows = (await db.execute(
        select(
            ApiUsageEvent.agent,
            func.count(ApiUsageEvent.id),
            func.coalesce(func.sum(ApiUsageEvent.input_tokens + ApiUsageEvent.output_tokens), 0),
            func.coalesce(func.sum(ApiUsageEvent.image_count), 0),
            func.coalesce(func.sum(ApiUsageEvent.cost_usd), 0),
        ).where(ApiUsageEvent.created_at >= start, ApiUsageEvent.created_at < end)
        .group_by(ApiUsageEvent.agent).order_by(func.sum(ApiUsageEvent.cost_usd).desc())
    )).all()
    agents = [
        {
            "agent": r[0],
            "calls": int(r[1] or 0),
            "tokens": int(r[2] or 0),
            "images": int(r[3] or 0),
            "cost_usd": float(r[4] or 0),
        }
        for r in rows
    ]
    return {**_window_payload(window), "agents": agents}


async def model_breakdown(
    db: AsyncSession,
    *,
    days: Optional[int] = None,
    range_key: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    window = await resolve_metrics_window(
        db, days=days, range_key=range_key, date_from=date_from, date_to=date_to
    )
    start, end = window["start"], window["end_exclusive"]
    rows = (await db.execute(
        select(
            ApiUsageEvent.model,
            ApiUsageEvent.provider,
            func.count(ApiUsageEvent.id),
            func.coalesce(func.sum(ApiUsageEvent.input_tokens + ApiUsageEvent.output_tokens), 0),
            func.coalesce(func.sum(ApiUsageEvent.image_count), 0),
            func.coalesce(func.sum(ApiUsageEvent.cost_usd), 0),
        ).where(ApiUsageEvent.created_at >= start, ApiUsageEvent.created_at < end)
        .group_by(ApiUsageEvent.model, ApiUsageEvent.provider).order_by(func.sum(ApiUsageEvent.cost_usd).desc())
    )).all()
    models = [
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
    return {**_window_payload(window), "models": models}


# ── Time on platform ──────────────────────────────────────────────────────────


async def time_spent(
    db: AsyncSession,
    *,
    days: Optional[int] = None,
    range_key: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    """Mean & median daily active time per user (minutes) + a daily series.

    Estimated from session_activity heartbeats: per (user, day), active minutes
    ≈ number of heartbeats (one per ~60s) — but capped and deduped to 60s
    intervals. We approximate as COUNT(DISTINCT DATE_TRUNC('minute', occurred_at)).
    The ``daily`` series aggregates those per-user-day minutes by calendar day so
    the admin UI can chart active minutes over time. Empty days are zero-filled.
    """
    window = await resolve_metrics_window(
        db, days=days, range_key=range_key, date_from=date_from, date_to=date_to
    )
    start, end = window["start"], window["end_exclusive"]
    sql = text("""
        SELECT user_id, day, COUNT(*) AS minutes
        FROM (
            SELECT
                user_id,
                DATE_TRUNC('day', occurred_at) AS day,
                DATE_TRUNC('minute', occurred_at) AS minute
            FROM session_activity
            WHERE occurred_at >= :start AND occurred_at < :end AND user_id IS NOT NULL
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
            WHERE occurred_at >= :start AND occurred_at < :end AND user_id IS NOT NULL
            GROUP BY user_id, day, minute
        ) t
        GROUP BY day
        ORDER BY day
    """)
    empty = {
        **_window_payload(window),
        "mean_minutes": 0,
        "median_minutes": 0,
        "sample_user_days": 0,
        "daily": fill_daily_series(
            window["start_day"],
            window["end_day"],
            {},
            defaults={"active_minutes": 0, "users": 0},
        ),
    }
    try:
        rows = (await db.execute(sql, {"start": start, "end": end})).all()
    except Exception:
        return empty
    minutes = [int(r.minutes or 0) for r in rows]
    if not minutes:
        return empty
    minutes.sort()
    n = len(minutes)
    median = minutes[n // 2] if n % 2 else (minutes[n // 2 - 1] + minutes[n // 2]) / 2

    by_day: dict[date, dict] = {}
    try:
        drows = (await db.execute(daily_sql, {"start": start, "end": end})).all()
        for r in drows:
            d = _as_utc_day(r.day)
            by_day[d] = {
                "active_minutes": int(r.minutes or 0),
                "users": int(r.users or 0),
            }
    except Exception:
        pass

    daily = fill_daily_series(
        window["start_day"],
        window["end_day"],
        by_day,
        defaults={"active_minutes": 0, "users": 0},
    )

    return {
        **_window_payload(window),
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
    window = resolve_date_window(days=30)
    start, end = window["start"], window["end_exclusive"]
    daily_rows = (await db.execute(
        select(
            func.date_trunc("day", ApiUsageEvent.created_at).label("d"),
            func.count(ApiUsageEvent.id),
            func.coalesce(func.sum(ApiUsageEvent.cost_usd), 0),
        ).where(
            ApiUsageEvent.user_id == user_id,
            ApiUsageEvent.created_at >= start,
            ApiUsageEvent.created_at < end,
        ).group_by("d").order_by("d")
    )).all()
    by_day = {
        _as_utc_day(r.d): {"calls": int(r[1] or 0), "cost_usd": float(r[2] or 0)}
        for r in daily_rows
    }
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
        "daily": fill_daily_series(
            window["start_day"],
            window["end_day"],
            by_day,
            defaults={"calls": 0, "cost_usd": 0.0},
        ),
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
