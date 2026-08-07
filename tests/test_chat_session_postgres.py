"""Postgres-backed chat_session isolation tests (fake shared store = multi-worker DB)."""
from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from app import chat_session
from app.db.models import ChatSessionRow, GuestSession


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class _FakeChatDb:
    """Shared dict store standing in for ``chat_sessions`` across workers."""

    by_user: dict[int, ChatSessionRow] = {}
    by_guest: dict[str, ChatSessionRow] = {}
    guest_sessions: dict[str, GuestSession] = {}
    _next_id = 1
    _next_guest_id = 1
    # When True, chat_sessions insert fails unless guest_sessions parent exists
    # (mirrors Postgres FK chat_sessions_guest_session_id_fkey).
    enforce_guest_fk: bool = True

    def __init__(self) -> None:
        self._pending: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def add(self, row) -> None:
        self._pending.append(row)

    async def flush(self) -> None:
        for row in self._pending:
            if isinstance(row, GuestSession):
                if row.session_id in _FakeChatDb.guest_sessions:
                    raise IntegrityError(
                        "duplicate guest_sessions.session_id",
                        params=None,
                        orig=Exception("unique"),
                    )
                if getattr(row, "id", None) is None:
                    row.id = _FakeChatDb._next_guest_id
                    _FakeChatDb._next_guest_id += 1
                _FakeChatDb.guest_sessions[row.session_id] = row
                continue

            if (
                _FakeChatDb.enforce_guest_fk
                and row.guest_session_id
                and row.guest_session_id not in _FakeChatDb.guest_sessions
            ):
                raise IntegrityError(
                    "chat_sessions_guest_session_id_fkey",
                    params=None,
                    orig=Exception("fk"),
                )
            if getattr(row, "id", None) is None:
                row.id = _FakeChatDb._next_id
                _FakeChatDb._next_id += 1
            if row.user_id is not None:
                _FakeChatDb.by_user[int(row.user_id)] = row
            if row.guest_session_id:
                _FakeChatDb.by_guest[row.guest_session_id] = row
        self._pending.clear()

    async def commit(self) -> None:
        await self.flush()

    async def rollback(self) -> None:
        self._pending.clear()

    async def delete(self, row: ChatSessionRow) -> None:
        if row.user_id is not None:
            _FakeChatDb.by_user.pop(int(row.user_id), None)
        if row.guest_session_id:
            _FakeChatDb.by_guest.pop(row.guest_session_id, None)

    async def execute(self, stmt):
        # GuestSession.id select used by _ensure_guest_session_exists
        cols = list(getattr(stmt, "_raw_columns", ()) or ())
        entity = None
        if cols:
            col0 = cols[0]
            entity = getattr(getattr(col0, "table", None), "name", None) or getattr(
                getattr(col0, "entity_namespace", None), "__tablename__", None
            )
            # select(GuestSession.id) → column table name guest_sessions
            parent = getattr(col0, "table", None)
            if parent is not None and getattr(parent, "name", None) == "guest_sessions":
                entity = "guest_sessions"
            key_name = getattr(col0, "key", None)
            if key_name == "id" and entity == "guest_sessions":
                criteria = list(getattr(stmt, "_where_criteria", ()) or ())
                for crit in criteria:
                    left = getattr(crit, "left", None)
                    right = getattr(crit, "right", None)
                    key = getattr(left, "key", None)
                    val = getattr(right, "value", right)
                    if key == "session_id":
                        guest = _FakeChatDb.guest_sessions.get(val)
                        return _FakeResult(guest.id if guest else None)

        criteria = list(getattr(stmt, "_where_criteria", ()) or ())
        for crit in criteria:
            left = getattr(crit, "left", None)
            right = getattr(crit, "right", None)
            key = getattr(left, "key", None)
            val = getattr(right, "value", right)
            table = getattr(getattr(left, "table", None), "name", None)
            if key == "session_id" and table == "guest_sessions":
                guest = _FakeChatDb.guest_sessions.get(val)
                return _FakeResult(guest.id if guest is not None else None)
            if key == "user_id":
                return _FakeResult(_FakeChatDb.by_user.get(int(val)))
            if key == "guest_session_id":
                return _FakeResult(_FakeChatDb.by_guest.get(val))
        return _FakeResult(None)


@pytest.fixture(autouse=True)
def fake_chat_db(monkeypatch):
    _FakeChatDb.by_user.clear()
    _FakeChatDb.by_guest.clear()
    _FakeChatDb.guest_sessions.clear()
    _FakeChatDb._next_id = 1
    _FakeChatDb._next_guest_id = 1
    _FakeChatDb.enforce_guest_fk = True

    def _maker():
        return _FakeChatDb()

    monkeypatch.setattr(chat_session, "async_session_maker", _maker)
    yield
    _FakeChatDb.by_user.clear()
    _FakeChatDb.by_guest.clear()
    _FakeChatDb.guest_sessions.clear()


def _run(coro):
    return asyncio.run(coro)


def test_chat_session_users_do_not_share_interaction_ids():
    _run(chat_session.set_interaction_id(1, "int-user-1"))
    _run(chat_session.set_interaction_id(2, "int-user-2"))
    assert _run(chat_session.get_interaction_id(1)) == "int-user-1"
    assert _run(chat_session.get_interaction_id(2)) == "int-user-2"


def test_chat_session_guests_do_not_share_interaction_ids():
    guest_a = str(uuid.uuid4())
    guest_b = str(uuid.uuid4())
    _run(chat_session.set_guest_interaction_id(guest_a, "int-guest-a"))
    _run(chat_session.set_guest_interaction_id(guest_b, "int-guest-b"))
    assert _run(chat_session.get_guest_interaction_id(guest_a)) == "int-guest-a"
    assert _run(chat_session.get_guest_interaction_id(guest_b)) == "int-guest-b"
    assert guest_a in _FakeChatDb.guest_sessions
    assert guest_b in _FakeChatDb.guest_sessions


def test_set_guest_interaction_id_ensures_missing_guest_session_row():
    """FK parent may be absent (never registered / rolled back / separate conn)."""
    guest_id = str(uuid.uuid4())
    assert guest_id not in _FakeChatDb.guest_sessions
    _run(chat_session.set_guest_interaction_id(guest_id, "int-new"))
    assert guest_id in _FakeChatDb.guest_sessions
    assert _run(chat_session.get_guest_interaction_id(guest_id)) == "int-new"


def test_append_guest_turn_ensures_missing_guest_session_row():
    guest_id = str(uuid.uuid4())
    _run(chat_session.append_guest_turn(guest_id, "hello", "hi"))
    assert guest_id in _FakeChatDb.guest_sessions
    seeds = _run(chat_session.get_guest_voice_seed_turns(guest_id))
    assert seeds[0]["parts"][0]["text"] == "hello"


def test_ensure_guest_session_integrity_error_race():
    """Concurrent creator wins unique(session_id); ensure recovers via re-select."""
    guest_id = str(uuid.uuid4())
    calls = {"n": 0}
    real_flush = _FakeChatDb.flush

    async def flaky_flush(self):
        calls["n"] += 1
        # First flush is the GuestSession ensure insert — simulate concurrent win.
        if calls["n"] == 1 and self._pending and isinstance(self._pending[0], GuestSession):
            _FakeChatDb.guest_sessions[guest_id] = GuestSession(
                id=99, session_id=guest_id, generation_count=0,
                chat_turn_count=0, voice_turn_count=0,
            )
            self._pending.clear()
            raise IntegrityError("duplicate", params=None, orig=Exception("unique"))
        return await real_flush(self)

    _FakeChatDb.flush = flaky_flush  # type: ignore[method-assign]
    try:
        _run(chat_session.set_guest_interaction_id(guest_id, "int-race"))
        assert _run(chat_session.get_guest_interaction_id(guest_id)) == "int-race"
    finally:
        _FakeChatDb.flush = real_flush  # type: ignore[method-assign]


def test_chat_session_transcripts_are_per_user():
    _run(chat_session.append_turn(10, "hello", "hi there"))
    _run(chat_session.append_turn(11, "other", "reply"))
    assert _run(chat_session.get_voice_seed_turns(10))[0]["parts"][0]["text"] == "hello"
    assert _run(chat_session.get_voice_seed_turns(11))[0]["parts"][0]["text"] == "other"


def test_clearing_one_user_does_not_affect_another():
    _run(chat_session.set_interaction_id(5, "keep-me"))
    _run(chat_session.set_interaction_id(6, "drop-me"))
    _run(chat_session.clear_session(6))
    assert _run(chat_session.get_interaction_id(5)) == "keep-me"
    assert _run(chat_session.get_interaction_id(6)) is None


def test_chat_session_survives_cross_worker_read():
    """Worker B with a fresh process still sees Worker A's Postgres row."""
    _run(chat_session.set_interaction_id(42, "shared-iid"))
    _run(chat_session.append_turn(42, "from A", "reply A"))
    # Simulate worker B: same shared store (fake DB), no in-process dicts.
    assert _run(chat_session.get_interaction_id(42)) == "shared-iid"
    seeds = _run(chat_session.get_voice_seed_turns(42))
    assert seeds[0]["role"] == "user"
    assert seeds[0]["parts"][0]["text"] == "from A"
