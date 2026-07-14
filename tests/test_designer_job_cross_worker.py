"""Deploy regression test: designer job state must survive a cross-worker hit.

Production runs gunicorn with multiple UvicornWorker. The create endpoint lands
on worker A (job lives in that process's in-memory ``JOBS`` registry), while the
poll/events endpoint frequently lands on worker B, whose ``JOBS`` is empty. The
fix persists job rows + the JSONB event log to Postgres so any worker can serve
GET /designer/jobs/{id} and /events.

This test simulates two workers against a *shared* fake DB (a module-level dict
standing in for the ``designer_jobs`` table) by swapping out
``app.designer_agent.async_session_maker``. We:

1. Worker A: create a job, persist it, emit a couple of progress events, then
   mark it terminal (completed) with a final payload.
2. Worker B: clear the in-memory ``JOBS`` registry (B doesn't own the job),
   then call ``get_job_for_user`` and drain ``stream_job_events_from_db``.

Both must succeed without a 404, and the events streamed to B must match what
worker A persisted. This is the exact regression that caused
GET /designer/jobs/{id} -> 404 in prod.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest


class _FakeRow(SimpleNamespace):
    """Mutable stand-in for a DesignerJobRow; attribute writes are persisted
    through the shared store because the same object reference is kept there."""


class _FakeDesignerSession:
    """Minimal AsyncSession shim backed by a shared dict (the 'designer_jobs'
    table). Implements just the surface the persistence helpers use:
    ``add`` / ``get`` / ``execute(text, params)`` / ``commit``."""

    store: dict[str, _FakeRow] = {}

    def __init__(self) -> None:
        self._pending: list[_FakeRow] = []

    async def __aenter__(self) -> "_FakeDesignerSession":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    def add(self, row: _FakeRow) -> None:
        self._pending.append(row)

    async def get(self, model, pk):
        return _FakeDesignerSession.store.get(pk)

    async def execute(self, stmt, params=None):
        jid = params.get("id") if params else None
        row = _FakeDesignerSession.store.get(jid) if jid else None
        if row is None:
            return None
        if params and "evt" in params:
            appended = json.loads(params["evt"])
            row.events = list(row.events or []) + appended
            if "status" in params:
                row.status = params["status"]
        elif params and "uid" in params:
            if row.user_id == params["uid"]:
                row.cancel_requested = True
        return None

    async def commit(self) -> None:
        for r in self._pending:
            _FakeDesignerSession.store[r.id] = r
        self._pending.clear()


@pytest.fixture
def fake_db(monkeypatch):
    _FakeDesignerSession.store.clear()

    def _maker():
        return _FakeDesignerSession()

    # Both the module-level name used by the helpers and any imported alias.
    monkeypatch.setattr("app.designer_agent.async_session_maker", _maker)
    yield _FakeDesignerSession.store
    _FakeDesignerSession.store.clear()


def _user(uid: int):
    return SimpleNamespace(id=uid)


async def _drain(stream, *, max_chunks: int = 50) -> list[str]:
    out: list[str] = []
    agen = stream
    if hasattr(agen, "__aiter__") and not hasattr(agen, "__anext__"):
        agen = agen.__aiter__()
    try:
        while len(out) < max_chunks:
            chunk = await asyncio.wait_for(agen.__anext__(), timeout=5)
            out.append(chunk)
    except (StopAsyncIteration, asyncio.TimeoutError):
        pass
    return out


def test_designer_job_survives_cross_worker_get(fake_db):
    """Worker B (empty JOBS) can GET a job created on worker A via the DB row."""
    from app import designer_agent as da

    user = _user(7)
    job = da.create_job(user)
    job.status = "running"
    asyncio.run(da._persist_job_create(job))

    # Simulate worker B: in-memory registry does NOT contain this job.
    da.JOBS.clear()

    fetched = asyncio.run(da.get_job_for_user(job.id, 7))
    assert fetched is not None, "cross-worker GET returned 404"
    assert fetched.id == job.id
    assert fetched.user_id == 7
    assert fetched.status == "running"

    # Wrong owner must still 404 (no leakage across users).
    assert asyncio.run(da.get_job_for_user(job.id, 999)) is None


def test_designer_job_cross_worker_events_replay(fake_db):
    """Worker B's /events stream replays persisted events and terminates."""
    from app import designer_agent as da

    user = _user(7)
    job = da.create_job(user)
    asyncio.run(da._persist_job_create(job))

    job.status = "running"
    asyncio.run(da.emit(job, "progress", message="step 1"))
    asyncio.run(da.emit(job, "progress", message="step 2"))

    job.status = "completed"
    job.final = {"best": {"id": "g1"}}
    asyncio.run(da._persist_job_final(job, final=job.final))

    # Worker B has no in-memory job.
    da.JOBS.clear()

    chunks = asyncio.run(_drain(da.stream_job_events_from_db(job.id, 7)))
    payload_blobs = [c for c in chunks if c.startswith("data: ")]
    assert len(payload_blobs) == 2, payload_blobs
    parsed = [json.loads(c[len("data: "):].strip()) for c in payload_blobs]
    assert parsed[0]["type"] == "progress"
    assert parsed[0]["message"] == "step 1"
    assert parsed[1]["message"] == "step 2"
    # Stream must terminate (terminal status) rather than hang for keep-alives.
    assert not any(c.startswith(": keep-alive") for c in chunks)


def test_designer_job_cross_worker_cancel_persists(fake_db):
    """A cancel request landing on worker B is durably recorded in the DB."""
    from app import designer_agent as da

    user = _user(7)
    job = da.create_job(user)
    asyncio.run(da._persist_job_create(job))

    # Worker B: doesn't own the job.
    da.JOBS.clear()

    ok = asyncio.run(da.cancel_job(job.id, 7))
    assert ok is True
    row = _FakeDesignerSession.store[job.id]
    assert row.cancel_requested is True

    # Cancel from the wrong user must fail and not flip the flag back.
    assert asyncio.run(da.cancel_job(job.id, 999)) is False


def test_designer_job_wrong_owner_get_returns_none(fake_db):
    """A user must never read another user's job, even via the DB fallback."""
    from app import designer_agent as da

    user = _user(7)
    job = da.create_job(user)
    asyncio.run(da._persist_job_create(job))
    da.JOBS.clear()

    assert asyncio.run(da.get_job_for_user(job.id, 8)) is None
    assert asyncio.run(da.get_job_for_user("nonexistent-id", 7)) is None
