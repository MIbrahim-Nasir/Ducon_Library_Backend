"""Unit tests for admin metrics date windows, zero-fill, and aggregation helpers."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.admin.metrics import (
    aggregate_series_by_week,
    fill_daily_series,
    resolve_date_window,
)


FIXED_NOW = datetime(2026, 7, 23, 15, 30, 0, tzinfo=timezone.utc)


def test_resolve_default_30d_ends_today_inclusive():
    w = resolve_date_window(now=FIXED_NOW)
    assert w["end_day"] == date(2026, 7, 23)
    assert w["start_day"] == date(2026, 6, 24)
    assert w["day_count"] == 30
    assert w["range"] == "30d"
    assert w["timezone"] == "UTC"
    assert w["start"] == datetime(2026, 6, 24, tzinfo=timezone.utc)
    assert w["end_exclusive"] == datetime(2026, 7, 24, tzinfo=timezone.utc)


def test_resolve_7d_and_90d_and_ytd():
    w7 = resolve_date_window(range_key="7d", now=FIXED_NOW)
    assert w7["start_day"] == date(2026, 7, 17)
    assert w7["day_count"] == 7

    w90 = resolve_date_window(range_key="90d", now=FIXED_NOW)
    assert w90["day_count"] == 90
    assert w90["start_day"] == date(2026, 4, 25)

    wy = resolve_date_window(range_key="ytd", now=FIXED_NOW)
    assert wy["start_day"] == date(2026, 1, 1)
    assert wy["end_day"] == date(2026, 7, 23)
    assert wy["day_count"] == (date(2026, 7, 23) - date(2026, 1, 1)).days + 1


def test_resolve_all_uses_earliest_and_clamps():
    earliest = date(2020, 1, 1)
    w = resolve_date_window(range_key="all", now=FIXED_NOW, earliest_day=earliest)
    assert w["end_day"] == date(2026, 7, 23)
    # clamped to max series length
    assert w["day_count"] == 1095
    assert w["start_day"] == date(2026, 7, 23) - __import__("datetime").timedelta(days=1094)


def test_resolve_custom_from_to_inclusive():
    w = resolve_date_window(date_from="2026-07-10", date_to="2026-07-12", now=FIXED_NOW)
    assert w["range"] == "custom"
    assert w["start_day"] == date(2026, 7, 10)
    assert w["end_day"] == date(2026, 7, 12)
    assert w["day_count"] == 3
    assert str(w["start"].date()) == "2026-07-10"
    assert w["end_exclusive"].date() == date(2026, 7, 13)


def test_resolve_custom_clamps_future_to_today():
    w = resolve_date_window(date_from="2026-07-20", date_to="2026-08-01", now=FIXED_NOW)
    assert w["end_day"] == date(2026, 7, 23)


def test_resolve_legacy_days_is_inclusive_calendar():
    w = resolve_date_window(days=1, now=FIXED_NOW)
    assert w["start_day"] == date(2026, 7, 23)
    assert w["end_day"] == date(2026, 7, 23)
    assert w["day_count"] == 1

    w30 = resolve_date_window(days=30, now=FIXED_NOW)
    assert w30["day_count"] == 30
    assert w30["end_day"] == date(2026, 7, 23)


def test_resolve_invalid_range():
    with pytest.raises(ValueError):
        resolve_date_window(range_key="nope", now=FIXED_NOW)


def test_fill_daily_series_zero_fills_gaps_through_end():
    by_day = {
        date(2026, 7, 17): {"active_users": 5, "new_users": 2},
        date(2026, 7, 20): {"active_users": 1},
    }
    series = fill_daily_series(
        date(2026, 7, 17),
        date(2026, 7, 23),
        by_day,
        defaults={"active_users": 0, "new_users": 0, "new_guests": 0},
    )
    assert len(series) == 7
    assert series[0] == {"day": "2026-07-17", "active_users": 5, "new_users": 2, "new_guests": 0}
    assert series[-1]["day"] == "2026-07-23"
    assert series[-1]["active_users"] == 0
    assert series[1]["day"] == "2026-07-18"
    assert series[1]["active_users"] == 0
    assert series[3]["day"] == "2026-07-20"
    assert series[3]["active_users"] == 1


def test_fill_preserves_trailing_zeros_so_charts_reach_today():
    """Regression: sparse DAU used to omit empty trailing days → chart ended early."""
    series = fill_daily_series(
        date(2026, 7, 1),
        date(2026, 7, 23),
        {date(2026, 7, 17): {"active_users": 3}},
        defaults={"active_users": 0},
    )
    assert series[-1]["day"] == "2026-07-23"
    assert series[-1]["active_users"] == 0
    assert len(series) == 23


def test_aggregate_series_by_week_sums_metrics():
    daily = fill_daily_series(
        date(2026, 7, 20),  # Monday
        date(2026, 7, 23),  # Thursday
        {
            date(2026, 7, 20): {"new_users": 1, "new_guests": 2},
            date(2026, 7, 22): {"new_users": 3, "new_guests": 0},
        },
        defaults={"new_users": 0, "new_guests": 0},
    )
    weekly = aggregate_series_by_week(daily, ["new_users", "new_guests"])
    assert len(weekly) == 1
    assert weekly[0]["week_start"] == "2026-07-20"
    assert weekly[0]["new_users"] == 4
    assert weekly[0]["new_guests"] == 2
