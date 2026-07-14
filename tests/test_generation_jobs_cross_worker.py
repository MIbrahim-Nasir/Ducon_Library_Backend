"""Generation jobs survive cross-worker GET + events (Postgres mirror)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest


class _FakeRow(SimpleNamespace):
    pass


class _FakeGenSession:
    store: dict[str, _FakeRow] = {}

    def __init__(self) -> None:
        self._pending: list[_FakeRow] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def add(self, row) -> None:
        # Convert ORM row to mutable SimpleNamespace for attribute writes.
        self._pending.append(
            _FakeRow(
                id=row.id,
                user_id=row.user_id,
                guest_session_id=row.guest_session_id,
                status=row.status,
                events=list(getattr(row, "events", None) or []),
                final=getattr(row, "final", None),
                error=getattr(row, "error", None),
                cancel_requested=bool(getattr(row, "cancel_requested", False)),
                request_id=getattr(row, "request_id", None),
                created_at=None,
                updated_at=None,
            )
        )

    async def get(self, model, pk):
        return _FakeGenSession.store.get(pk)

    async def execute(self, stmt, params=None):
        jid = params.get("id") if params else None
        row = _FakeGenSession.store.get(jid) if jid else None
        if row is None:
            return None
        if params and "evt" in params:
            appended = json.loads(params["evt"])
            row.events = list(row.events or []) + appended
            if "status" in params:
                row.status = params["status"]
        elif params and ("uid" in params or "gid" in params):
            row.cancel_requested = True
        return None

    async def commit(self) -> None:
        for r in self._pending:
            _FakeGenSession.store[r.id] = r
        self._pending.clear()


@pytest.fixture
def fake_db(monkeypatch):
    _FakeGenSession.store.clear()

    def _maker():
        return _FakeGenSession()

    monkeypatch.setattr("app.generation_jobs.async_session_maker", _maker)
    yield _FakeGenSession.store
    _FakeGenSession.store.clear()


async def _drain(stream, *, max_chunks: int = 50) -> list[str]:
    out: list[str] = []
    agen = stream.__aiter__() if hasattr(stream, "__aiter__") else stream
    try:
        while len(out) < max_chunks:
            chunk = await asyncio.wait_for(agen.__anext__(), timeout=5)
            out.append(chunk)
    except (StopAsyncIteration, asyncio.TimeoutError):
        pass
    return out


def test_generation_job_survives_cross_worker_get(fake_db):
    from app import generation_jobs as gj

    job = gj.create_job(user_id=7, request_id="rid-1")
    job.status = "running"
    asyncio.run(gj.persist_job_create(job))

    gj.JOBS.clear()  # worker B

    fetched = asyncio.run(gj.get_job_for_owner(job.id, user_id=7))
    assert fetched is not None
    assert fetched.id == job.id
    assert fetched.user_id == 7
    assert fetched.request_id == "rid-1"
    assert asyncio.run(gj.get_job_for_owner(job.id, user_id=999)) is None


def test_generation_job_cross_worker_events_replay(fake_db):
    from app import generation_jobs as gj

    job = gj.create_job(user_id=7)
    asyncio.run(gj.persist_job_create(job))
    job.status = "running"
    asyncio.run(gj.emit(job, "status", message="started"))
    job.status = "completed"
    job.final = {"id": 1, "url": "https://example.test/g.png"}
    asyncio.run(gj.emit(job, "done", **job.final))
    asyncio.run(gj._persist_job_final(job, final=job.final))

    gj.JOBS.clear()
    chunks = asyncio.run(_drain(gj.stream_job_events_from_db(job.id, user_id=7)))
    joined = "".join(chunks)
    assert "started" in joined
    assert "done" in joined or '"type": "done"' in joined or "example.test" in joined


def test_generation_job_guest_ownership(fake_db):
    from app import generation_jobs as gj

    gid = "11111111-1111-1111-1111-111111111111"
    job = gj.create_job(guest_session_id=gid)
    asyncio.run(gj.persist_job_create(job))
    gj.JOBS.clear()
    assert asyncio.run(gj.get_job_for_owner(job.id, guest_session_id=gid)) is not None
    assert asyncio.run(gj.get_job_for_owner(job.id, guest_session_id="other")) is None
    assert asyncio.run(gj.get_job_for_owner(job.id, user_id=1)) is None
