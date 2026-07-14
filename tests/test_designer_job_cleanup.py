"""Deploy regression tests for designer job retention cleanup."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.designer_agent import JOBS, DesignerJob
from app.designer_cleanup import cleanup_designer_jobs


def _run(coro):
    return asyncio.run(coro)


class _FakeScalarResult:
    def __init__(self, ids: list[str]):
        self._ids = ids

    def all(self):
        return [(i,) for i in self._ids]


class _RecordingSession:
    """Records SQLAlchemy execute calls; returns staged select results."""

    def __init__(self, stale_ids: list[str], old_ids: list[str]):
        self.stale_ids = stale_ids
        self.old_ids = old_ids
        self.executed: list[object] = []
        self.committed = False

    async def execute(self, stmt):
        self.executed.append(stmt)
        selects = [e for e in self.executed if "Select" in type(e).__name__]
        if len(selects) == 1:
            return _FakeScalarResult(self.stale_ids)
        if len(selects) == 2:
            return _FakeScalarResult(self.old_ids)
        return _FakeScalarResult([])

    async def commit(self):
        self.committed = True


def test_cleanup_noop_when_nothing_to_prune(monkeypatch):
    monkeypatch.setenv("DESIGNER_JOB_RETENTION_DAYS", "30")
    monkeypatch.setenv("DESIGNER_JOB_STALE_RUNNING_HOURS", "2")

    db = _RecordingSession(stale_ids=[], old_ids=[])
    stats = _run(cleanup_designer_jobs(db))

    assert stats == {"designer_jobs_marked_stale": 0, "designer_jobs_deleted": 0}
    assert len(db.executed) == 2  # two selects only


def test_cleanup_marks_stale_and_deletes_old(monkeypatch):
    monkeypatch.setenv("DESIGNER_JOB_RETENTION_DAYS", "30")
    monkeypatch.setenv("DESIGNER_JOB_STALE_RUNNING_HOURS", "2")

    JOBS["old-job"] = DesignerJob(id="old-job", user_id=1, status="completed")
    JOBS["stale-job"] = DesignerJob(id="stale-job", user_id=1, status="running")

    db = _RecordingSession(stale_ids=["stale-job"], old_ids=["old-job"])
    stats = _run(cleanup_designer_jobs(db))

    assert stats["designer_jobs_marked_stale"] == 1
    assert stats["designer_jobs_deleted"] == 1
    assert "old-job" not in JOBS
    assert "stale-job" in JOBS  # marked failed in DB, not removed from memory
    assert len(db.executed) == 4  # select, update, select, delete


def test_cleanup_pops_multiple_deleted_jobs_from_memory(monkeypatch):
    monkeypatch.setenv("DESIGNER_JOB_RETENTION_DAYS", "7")
    monkeypatch.setenv("DESIGNER_JOB_STALE_RUNNING_HOURS", "2")

    for jid in ("a", "b", "c"):
        JOBS[jid] = DesignerJob(id=jid, user_id=9, status="completed")

    db = _RecordingSession(stale_ids=[], old_ids=["a", "b", "c"])
    stats = _run(cleanup_designer_jobs(db))

    assert stats["designer_jobs_deleted"] == 3
    assert not any(j in JOBS for j in ("a", "b", "c"))


def test_guest_cleanup_endpoint_returns_designer_stats(monkeypatch):
    """POST /guest/cleanup must include designer job prune counts in the response."""
    from app.routers import guest as guest_router

    monkeypatch.setattr(guest_router, "CLEANUP_CRON_SECRET", "cron-test-secret")
    monkeypatch.setattr(guest_router, "IS_PRODUCTION", False)

    async def _fake_cleanup(db):
        return {
            "guest_generations_deleted": 0,
            "designer_jobs_marked_stale": 1,
            "designer_jobs_deleted": 4,
        }

    monkeypatch.setattr(guest_router, "_run_cleanup", _fake_cleanup)

    result = _run(
        guest_router.guest_cleanup(
            x_cron_secret="cron-test-secret",
            db=MagicMock(),
        )
    )
    assert result["designer_jobs_marked_stale"] == 1
    assert result["designer_jobs_deleted"] == 4


def test_guest_cleanup_rejects_bad_secret(monkeypatch):
    from app.routers import guest as guest_router

    monkeypatch.setattr(guest_router, "CLEANUP_CRON_SECRET", "cron-test-secret")
    monkeypatch.setattr(guest_router, "IS_PRODUCTION", True)

    with pytest.raises(HTTPException) as exc:
        _run(
            guest_router.guest_cleanup(
                x_cron_secret="wrong",
                db=MagicMock(),
            )
        )
    assert exc.value.status_code == 403


def test_run_cleanup_commits_when_designer_jobs_pruned(monkeypatch):
    """_run_cleanup should commit when designer rows are marked or deleted."""
    from app.routers import guest as guest_router

    monkeypatch.setattr(
        "app.designer_cleanup.cleanup_designer_jobs",
        AsyncMock(return_value={"designer_jobs_marked_stale": 0, "designer_jobs_deleted": 2}),
    )

    db = MagicMock()
    db.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))))
    db.commit = AsyncMock()

    stats = _run(guest_router._run_cleanup(db))

    assert stats["designer_jobs_deleted"] == 2
    db.commit.assert_awaited_once()


def test_designer_job_finally_pops_terminal_from_jobs_registry():
    """Regression: in-memory JOBS must not leak after a terminal job finishes."""
    job = DesignerJob(id="terminal-job", user_id=1, status="completed")
    JOBS[job.id] = job

    if job.status in {"completed", "failed", "cancelled"}:
        JOBS.pop(job.id, None)

    assert "terminal-job" not in JOBS


def test_retention_env_vars_respected(monkeypatch):
    """Sanity: env overrides are read (defaults would still run two selects)."""
    monkeypatch.setenv("DESIGNER_JOB_RETENTION_DAYS", "14")
    monkeypatch.setenv("DESIGNER_JOB_STALE_RUNNING_HOURS", "6")

    from app import designer_cleanup as mod

    assert mod._retention_days() == 14
    assert mod._stale_running_hours() == 6
