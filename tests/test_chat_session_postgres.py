"""Postgres-backed chat_session isolation tests (fake shared store = multi-worker DB)."""
from __future__ import annotations

import asyncio
import uuid

import pytest

from app import chat_session
from app.db.models import ChatSessionRow


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class _FakeChatDb:
    """Shared dict store standing in for ``chat_sessions`` across workers."""

    by_user: dict[int, ChatSessionRow] = {}
    by_guest: dict[str, ChatSessionRow] = {}
    _next_id = 1

    def __init__(self) -> None:
        self._pending: list[ChatSessionRow] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def add(self, row: ChatSessionRow) -> None:
        self._pending.append(row)

    async def flush(self) -> None:
        for row in self._pending:
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

    async def delete(self, row: ChatSessionRow) -> None:
        if row.user_id is not None:
            _FakeChatDb.by_user.pop(int(row.user_id), None)
        if row.guest_session_id:
            _FakeChatDb.by_guest.pop(row.guest_session_id, None)

    async def execute(self, stmt):
        criteria = list(getattr(stmt, "_where_criteria", ()) or ())
        for crit in criteria:
            left = getattr(crit, "left", None)
            right = getattr(crit, "right", None)
            key = getattr(left, "key", None)
            val = getattr(right, "value", right)
            if key == "user_id":
                return _FakeResult(_FakeChatDb.by_user.get(int(val)))
            if key == "guest_session_id":
                return _FakeResult(_FakeChatDb.by_guest.get(val))
        return _FakeResult(None)


@pytest.fixture(autouse=True)
def fake_chat_db(monkeypatch):
    _FakeChatDb.by_user.clear()
    _FakeChatDb.by_guest.clear()
    _FakeChatDb._next_id = 1

    def _maker():
        return _FakeChatDb()

    monkeypatch.setattr(chat_session, "async_session_maker", _maker)
    yield
    _FakeChatDb.by_user.clear()
    _FakeChatDb.by_guest.clear()


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
