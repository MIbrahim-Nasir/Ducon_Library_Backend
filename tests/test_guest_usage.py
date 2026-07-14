"""Guest usage limits — per-device fingerprint identity, optional IP cap."""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.guest_usage import (
    GuestUsageKind,
    _validate_fingerprint,
    enforce_guest_limit,
    get_guest_session_row,
)
from app.hashing import sha256_hex


def _run(coro):
    return asyncio.run(coro)


VALID_FP = "a" * 64  # 64-char hex string (client header value)
STORED_FP = sha256_hex(VALID_FP)  # server-side stored hash


@pytest.fixture
def same_ip_hash():
    return "deadbeef" * 4


@pytest.fixture
def mock_db():
    return AsyncMock()


# ── Fingerprint validation ────────────────────────────────────────────────────

def test_validate_fingerprint_accepts_valid_sha256():
    assert _validate_fingerprint("a" * 64) == "a" * 64


def test_validate_fingerprint_rejects_short():
    assert _validate_fingerprint("abc") is None


def test_validate_fingerprint_rejects_uppercase():
    assert _validate_fingerprint("A" * 64) is None


def test_validate_fingerprint_rejects_none():
    assert _validate_fingerprint(None) is None


def test_get_guest_session_row_create_false_returns_none():
    """Read-only polls must not mint a guest_sessions row."""
    async def fake_execute(stmt):
        result = AsyncMock()
        result.scalar_one_or_none = lambda: None
        return result

    db = AsyncMock()
    db.execute = fake_execute
    db.add = AsyncMock()

    result = _run(
        get_guest_session_row(
            db,
            str(uuid.uuid4()),
            "abc",
            fingerprint_hash=None,
            create=False,
        )
    )
    assert result is None
    assert db.add.call_count == 0


# ── Fingerprint-first identity: UUID rotation is blocked ─────────────────────

def test_fingerprint_lookup_returns_existing_session_regardless_of_uuid(monkeypatch):
    """Same device clears localStorage → sends new UUID but same fingerprint → old session returned."""
    original_session = SimpleNamespace(
        id=1,
        session_id="original-uuid",
        ip_hash="abc",
        fingerprint_hash=STORED_FP,
        generation_count=2,   # already used 2 of 3
        chat_turn_count=0,
        voice_turn_count=0,
    )
    fresh_uuid = str(uuid.uuid4())

    execute_calls = []

    async def fake_execute(stmt):
        execute_calls.append(stmt)
        # First select is fingerprint lookup → return existing session
        if len(execute_calls) == 1:
            result = AsyncMock()
            result.scalar_one_or_none = lambda: original_session
            return result
        # Should not reach UUID lookup
        raise AssertionError("UUID lookup should not run when fingerprint matched")

    db = AsyncMock()
    db.execute = fake_execute

    result = _run(get_guest_session_row(db, fresh_uuid, "abc", fingerprint_hash=STORED_FP))
    assert result is original_session
    assert len(execute_calls) == 1  # only fingerprint query ran


def test_uuid_rotation_does_not_reset_quota(monkeypatch, mock_db):
    """With fingerprint: a fresh UUID still hits the exhausted original session → blocked."""
    monkeypatch.setattr("app.guest_usage._guest_ip_total_limit", lambda: 0)
    monkeypatch.setattr("app.guest_usage._guest_gen_limit", lambda: 3)

    exhausted_session = SimpleNamespace(
        id=1,
        session_id="original-uuid",
        ip_hash="abc",
        fingerprint_hash=STORED_FP,
        generation_count=3,  # limit exhausted
        chat_turn_count=0,
        voice_turn_count=0,
    )

    async def fake_get_row(db, session_id, ip_hash, *, fingerprint_hash=None, **kwargs):
        # Fingerprint match → same exhausted session regardless of session_id
        return exhausted_session

    monkeypatch.setattr("app.guest_usage.get_guest_session_row", fake_get_row)

    with pytest.raises(HTTPException) as exc:
        _run(
            enforce_guest_limit(
                mock_db,
                str(uuid.uuid4()),   # brand new UUID
                "abc",
                GuestUsageKind.GENERATION,
                fingerprint_hash=STORED_FP,
            )
        )
    assert exc.value.status_code == 429
    assert exc.value.detail["code"] == "GUEST_LIMIT_REACHED"


def test_invalid_fingerprint_skips_fingerprint_lookup():
    """Absent fingerprint_hash skips fingerprint lookup — only UUID lookup runs."""
    session_uuid = str(uuid.uuid4())
    session = SimpleNamespace(
        id=1,
        session_id=session_uuid,
        ip_hash="abc",
        fingerprint_hash=None,
        composite_hash=None,
        subnet_key=None,
        ja4_fingerprint=None,
        asn=None,
        generation_count=0,
        chat_turn_count=0,
        voice_turn_count=0,
    )
    execute_calls = []

    async def fake_execute(stmt):
        execute_calls.append(stmt)
        result = AsyncMock()
        result.scalar_one_or_none = lambda: session
        return result

    db = AsyncMock()
    db.execute = fake_execute
    db.flush = AsyncMock()

    result = _run(get_guest_session_row(db, session_uuid, "abc", fingerprint_hash=None))
    assert result is session
    assert len(execute_calls) == 1


def test_binds_fingerprint_to_existing_session_without_stored_hash():
    """UUID-only session gets fingerprint_hash bound on first fingerprinted request."""
    session_uuid = str(uuid.uuid4())
    existing = SimpleNamespace(
        id=5,
        session_id=session_uuid,
        ip_hash="abc",
        fingerprint_hash=None,
        composite_hash=None,
        subnet_key=None,
        ja4_fingerprint=None,
        asn=None,
        generation_count=0,
        chat_turn_count=0,
        voice_turn_count=0,
    )
    execute_calls = []

    async def fake_execute(stmt):
        execute_calls.append(stmt)
        result = AsyncMock()
        if len(execute_calls) == 1:
            result.scalar_one_or_none = lambda: None  # fingerprint miss
        elif len(execute_calls) == 2:
            result.scalar_one_or_none = lambda: existing  # UUID hit
        else:
            result.scalar_one_or_none = lambda: None  # update
        return result

    db = AsyncMock()
    db.execute = fake_execute
    db.flush = AsyncMock()

    result = _run(get_guest_session_row(db, session_uuid, "abc", fingerprint_hash=STORED_FP))
    assert result is existing
    assert existing.fingerprint_hash == STORED_FP
    assert len(execute_calls) == 3


def test_creates_session_with_fingerprint_when_no_match():
    """New guest row stores fingerprint_hash when device is first seen."""
    session_uuid = str(uuid.uuid4())
    execute_calls = []
    added = []

    async def fake_execute(stmt):
        execute_calls.append(stmt)
        result = AsyncMock()
        result.scalar_one_or_none = lambda: None
        return result

    db = AsyncMock()
    db.execute = fake_execute
    db.add = lambda row: added.append(row)
    db.flush = AsyncMock()

    _run(get_guest_session_row(db, session_uuid, "abc", fingerprint_hash=STORED_FP))
    assert len(execute_calls) == 2  # fingerprint lookup, then UUID lookup
    assert len(added) == 1
    assert added[0].session_id == session_uuid
    assert added[0].fingerprint_hash == STORED_FP
    assert added[0].generation_count == 0


def test_no_fingerprint_falls_back_to_uuid_lookup(monkeypatch, mock_db):
    """Without fingerprint header the lookup falls through to the UUID path."""
    monkeypatch.setattr("app.guest_usage._guest_ip_total_limit", lambda: 0)
    monkeypatch.setattr("app.guest_usage._guest_gen_limit", lambda: 3)

    session = SimpleNamespace(
        id=2,
        session_id="some-uuid",
        ip_hash="def",
        fingerprint_hash=None,
        generation_count=0,
        chat_turn_count=0,
        voice_turn_count=0,
    )

    async def fake_get_row(db, session_id, ip_hash, *, fingerprint_hash=None, **kwargs):
        assert fingerprint_hash is None
        return session

    monkeypatch.setattr("app.guest_usage.get_guest_session_row", fake_get_row)

    result = _run(
        enforce_guest_limit(
            mock_db,
            "some-uuid",
            "def",
            GuestUsageKind.GENERATION,
            fingerprint_hash=None,
        )
    )
    assert result is session


# ── Same-network coworkers stay independent ───────────────────────────────────

def test_same_network_different_sessions_allowed_when_ip_cap_disabled(
    monkeypatch, mock_db, same_ip_hash,
):
    """Two guest UUIDs on one office IP each get their own quota (default IP cap off)."""
    monkeypatch.setattr("app.guest_usage._guest_ip_total_limit", lambda: 0)
    monkeypatch.setattr("app.guest_usage._guest_subnet_limit", lambda: 0)
    monkeypatch.setattr("app.guest_usage._guest_gen_limit", lambda: 3)

    session_a = SimpleNamespace(
        id=1,
        session_id=str(uuid.uuid4()),
        ip_hash=same_ip_hash,
        fingerprint_hash=sha256_hex("device-a"),
        generation_count=3,
        chat_turn_count=0,
        voice_turn_count=0,
    )
    session_b = SimpleNamespace(
        id=2,
        session_id=str(uuid.uuid4()),
        ip_hash=same_ip_hash,
        fingerprint_hash=sha256_hex("device-b"),
        generation_count=0,
        chat_turn_count=0,
        voice_turn_count=0,
    )

    async def fake_get_row(db, session_id, ip_hash, *, fingerprint_hash=None, **kwargs):
        if fingerprint_hash == session_a.fingerprint_hash:
            return session_a
        return session_b

    monkeypatch.setattr("app.guest_usage.get_guest_session_row", fake_get_row)

    with pytest.raises(HTTPException) as blocked:
        _run(
            enforce_guest_limit(
                mock_db,
                session_a.session_id,
                same_ip_hash,
                GuestUsageKind.GENERATION,
                fingerprint_hash=session_a.fingerprint_hash,
            )
        )
    assert blocked.value.status_code == 429

    result = _run(
        enforce_guest_limit(
            mock_db,
            session_b.session_id,
            same_ip_hash,
            GuestUsageKind.GENERATION,
            fingerprint_hash=session_b.fingerprint_hash,
        )
    )
    assert result is session_b


# ── Optional IP cap ───────────────────────────────────────────────────────────

def test_shared_ip_cap_blocks_when_enabled(monkeypatch, mock_db, same_ip_hash):
    monkeypatch.setattr("app.guest_usage._guest_ip_total_limit", lambda: 15)

    async def fake_ip_total(db, ip_hash):
        return 15

    monkeypatch.setattr("app.guest_usage._ip_total_usage", fake_ip_total)
    monkeypatch.setattr(
        "app.guest_usage.get_guest_session_row",
        AsyncMock(side_effect=AssertionError("should not load session when IP cap blocks")),
    )

    with pytest.raises(HTTPException) as exc:
        _run(
            enforce_guest_limit(
                mock_db,
                str(uuid.uuid4()),
                same_ip_hash,
                GuestUsageKind.GENERATION,
            )
        )
    assert exc.value.status_code == 429
    assert exc.value.detail["code"] == "GUEST_LIMIT_REACHED"


def test_ip_cap_zero_never_blocks_even_with_high_usage(monkeypatch, mock_db):
    """Regression: GUEST_IP_TOTAL_LIMIT=0 must not block everyone (>= 0 was always true)."""
    monkeypatch.setattr("app.guest_usage._guest_ip_total_limit", lambda: 0)
    monkeypatch.setattr("app.guest_usage._guest_gen_limit", lambda: 3)

    session = SimpleNamespace(
        id=1,
        session_id=str(uuid.uuid4()),
        ip_hash="abc",
        fingerprint_hash=None,
        generation_count=0,
        chat_turn_count=0,
        voice_turn_count=0,
    )

    monkeypatch.setattr(
        "app.guest_usage.get_guest_session_row",
        AsyncMock(return_value=session),
    )

    result = _run(
        enforce_guest_limit(
            mock_db,
            session.session_id,
            "abc",
            GuestUsageKind.GENERATION,
        )
    )
    assert result is session


# ── Per-session limit still enforced ─────────────────────────────────────────

def test_session_generation_limit_blocks_per_guest(monkeypatch, mock_db, same_ip_hash):
    monkeypatch.setattr("app.guest_usage._guest_ip_total_limit", lambda: 0)
    monkeypatch.setattr("app.guest_usage._guest_gen_limit", lambda: 3)

    exhausted = SimpleNamespace(
        id=1,
        session_id=str(uuid.uuid4()),
        ip_hash=same_ip_hash,
        fingerprint_hash=None,
        generation_count=3,
        chat_turn_count=0,
        voice_turn_count=0,
    )

    async def fake_get_row(db, session_id, ip_hash, *, fingerprint_hash=None, **kwargs):
        return exhausted

    monkeypatch.setattr("app.guest_usage.get_guest_session_row", fake_get_row)

    with pytest.raises(HTTPException) as exc:
        _run(
            enforce_guest_limit(
                mock_db,
                exhausted.session_id,
                same_ip_hash,
                GuestUsageKind.GENERATION,
            )
        )
    assert exc.value.status_code == 429


# ── Router integration: get_or_create_guest_session ───────────────────────────

def test_get_or_create_guest_session_hashes_fingerprint(monkeypatch, mock_db):
    """X-Guest-Fingerprint header value is SHA-256 hashed before storage lookup."""
    from app.guest_identity import build_guest_request_identity
    from app.routers.guest import get_or_create_guest_session

    raw_fp = "b" * 64
    expected_hash = sha256_hex(raw_fp)
    session = SimpleNamespace(
        id=1,
        session_id=str(uuid.uuid4()),
        ip_hash=sha256_hex("127.0.0.1"),
        fingerprint_hash=expected_hash,
        generation_count=0,
        chat_turn_count=0,
        voice_turn_count=0,
    )
    identity = build_guest_request_identity(
        {"cf-connecting-ip": "127.0.0.1"},
        peer_host="127.0.0.1",
        raw_fingerprint=raw_fp,
    )

    async def fake_enforce(db, session_id, ip_hash, kind, *, fingerprint_hash=None, **kwargs):
        assert fingerprint_hash == expected_hash
        assert ip_hash == sha256_hex("127.0.0.1")
        return session

    monkeypatch.setattr("app.routers.guest.enforce_guest_limit", fake_enforce)

    result = _run(
        get_or_create_guest_session(mock_db, session.session_id, identity=identity)
    )
    assert result is session


def test_get_or_create_guest_session_blank_fingerprint_omitted(monkeypatch, mock_db):
    from app.guest_identity import build_guest_request_identity
    from app.routers.guest import get_or_create_guest_session

    session_id = str(uuid.uuid4())
    identity = build_guest_request_identity(
        {"cf-connecting-ip": "127.0.0.1"},
        peer_host="127.0.0.1",
        raw_fingerprint="   ",
    )

    async def fake_enforce(db, session_id_arg, ip_hash, kind, *, fingerprint_hash=None, **kwargs):
        assert session_id_arg == session_id
        assert fingerprint_hash is None
        return SimpleNamespace(session_id=session_id)

    monkeypatch.setattr("app.routers.guest.enforce_guest_limit", fake_enforce)

    _run(get_or_create_guest_session(mock_db, session_id, identity=identity))


def test_get_or_create_guest_session_rejects_invalid_uuid(mock_db):
    from app.guest_identity import build_guest_request_identity
    from app.routers.guest import get_or_create_guest_session

    with pytest.raises(HTTPException) as exc:
        _run(
            get_or_create_guest_session(
                mock_db,
                "not-a-uuid",
                identity=build_guest_request_identity({}),
            )
        )
    assert exc.value.status_code == 400
    assert "X-Guest-Session-Id" in exc.value.detail
