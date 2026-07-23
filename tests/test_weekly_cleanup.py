"""Tests for weekly retention cleanup (age filter + single-flight lock)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from app.cleanup_service import min_age_days, run_cleanup


def _run(coro):
    return asyncio.run(coro)


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Result:
    def __init__(self, *, rows=None, rowcount=0):
        self._rows = rows or []
        self.rowcount = rowcount

    def scalars(self):
        return _Scalars(self._rows)

    def all(self):
        return [(r,) for r in self._rows]


class _CleanupSession:
    """Minimal async session that stages guest select then empty/no-op results."""

    def __init__(self, guest_rows: list):
        self.guest_rows = guest_rows
        self.executed = []
        self.committed = False
        self._select_n = 0

    async def execute(self, stmt):
        self.executed.append(stmt)
        name = type(stmt).__name__
        if "Select" in name and self._select_n == 0:
            self._select_n += 1
            return _Result(rows=self.guest_rows)
        if "Select" in name:
            self._select_n += 1
            return _Result(rows=[])
        return _Result(rowcount=0)

    async def commit(self):
        self.committed = True


def test_min_age_days_default(monkeypatch):
    monkeypatch.setattr(
        "app.cleanup_service.cfg_int",
        lambda key, default=0: 7 if key == "CLEANUP_MIN_AGE_DAYS" else default,
    )
    assert min_age_days() == 7


def test_min_age_days_env_override(monkeypatch):
    monkeypatch.setattr(
        "app.cleanup_service.cfg_int",
        lambda key, default=0: 14 if key == "CLEANUP_MIN_AGE_DAYS" else default,
    )
    assert min_age_days() == 14


def test_run_cleanup_deletes_guest_older_than_min_age(monkeypatch):
    monkeypatch.setattr(
        "app.cleanup_service.cfg_int",
        lambda key, default=0: 7 if key == "CLEANUP_MIN_AGE_DAYS" else default,
    )
    monkeypatch.setattr(
        "app.designer_cleanup.cleanup_designer_jobs",
        AsyncMock(return_value={"designer_jobs_marked_stale": 0, "designer_jobs_deleted": 0}),
    )
    monkeypatch.setattr("app.cleanup_service.storage.delete_guest_generation", MagicMock())

    now = datetime.now(timezone.utc)
    old = MagicMock()
    old.id = 1
    old.url = "guests/old.jpg"
    old.generated_at = now - timedelta(days=10)
    old.expires_at = now - timedelta(days=8)
    old.user_id = None

    db = _CleanupSession(guest_rows=[old])
    stats = _run(run_cleanup(db, commit=True))

    assert stats["guest_generations_deleted"] == 1
    assert db.committed is True
    guest_select = db.executed[0]
    compiled = str(guest_select.compile(compile_kwargs={"literal_binds": False}))
    assert "generated_at" in compiled.lower()


def test_run_cleanup_keeps_when_select_returns_empty_young(monkeypatch):
    """Age filter is in SQL; empty select means nothing younger than cutoff is deleted."""
    monkeypatch.setattr(
        "app.cleanup_service.cfg_int",
        lambda key, default=0: 7 if key == "CLEANUP_MIN_AGE_DAYS" else default,
    )
    monkeypatch.setattr(
        "app.designer_cleanup.cleanup_designer_jobs",
        AsyncMock(return_value={"designer_jobs_marked_stale": 0, "designer_jobs_deleted": 0}),
    )

    db = _CleanupSession(guest_rows=[])
    stats = _run(run_cleanup(db, commit=True))

    assert stats["guest_generations_deleted"] == 0
    assert db.committed is False


def test_run_cleanup_includes_otp_and_jti_counts(monkeypatch):
    monkeypatch.setattr(
        "app.cleanup_service.cfg_int",
        lambda key, default=0: 7 if key == "CLEANUP_MIN_AGE_DAYS" else default,
    )
    monkeypatch.setattr(
        "app.designer_cleanup.cleanup_designer_jobs",
        AsyncMock(return_value={"designer_jobs_marked_stale": 0, "designer_jobs_deleted": 0}),
    )

    db = _CleanupSession(guest_rows=[])
    stats = _run(run_cleanup(db, commit=False))

    assert stats["revoked_jtis_deleted"] == 0
    assert stats["email_otps_deleted"] == 0


def test_scheduled_tick_skips_when_lock_held(monkeypatch):
    from app import cleanup_scheduler as sched
    import app.db.database as database

    monkeypatch.setattr(sched, "cleanup_enabled", lambda: True)
    monkeypatch.setattr(
        sched,
        "run_cleanup",
        AsyncMock(side_effect=AssertionError("cleanup must not run when lock held")),
    )

    class _Txn:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _LockSession:
        def begin(self):
            return _Txn()

        async def scalar(self, stmt):
            return False

        async def execute(self, *a, **k):
            raise AssertionError("unexpected execute while lock skipped")

    class _SessionCM:
        async def __aenter__(self):
            return _LockSession()

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(database, "async_session_maker", lambda: _SessionCM())

    result = _run(sched.run_scheduled_cleanup_tick())
    assert result["status"] == "lock_skipped"


def test_scheduled_tick_disabled(monkeypatch):
    from app import cleanup_scheduler as sched

    monkeypatch.setattr(sched, "cleanup_enabled", lambda: False)
    result = _run(sched.run_scheduled_cleanup_tick())
    assert result["status"] == "disabled"


def test_guest_endpoint_still_delegates(monkeypatch):
    from app.routers import guest as guest_router

    monkeypatch.setattr(guest_router, "CLEANUP_CRON_SECRET", "cron-test-secret")
    monkeypatch.setattr(guest_router, "IS_PRODUCTION", False)

    async def _fake(db):
        return {
            "guest_generations_deleted": 1,
            "designer_jobs_deleted": 0,
            "designer_jobs_marked_stale": 0,
            "generation_jobs_marked_stale": 0,
            "revoked_jtis_deleted": 2,
            "email_otps_deleted": 3,
        }

    monkeypatch.setattr(guest_router, "_run_cleanup", _fake)
    result = _run(
        guest_router.guest_cleanup(x_cron_secret="cron-test-secret", db=MagicMock())
    )
    assert result["guest_generations_deleted"] == 1
    assert result["email_otps_deleted"] == 3


def test_designer_retention_default_is_seven(monkeypatch):
    monkeypatch.delenv("DESIGNER_JOB_RETENTION_DAYS", raising=False)
    from app import designer_cleanup as mod

    assert mod._retention_days() == 7
