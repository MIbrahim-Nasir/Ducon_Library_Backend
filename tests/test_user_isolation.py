"""Global vs per-user isolation — rate limiter keys, chat sessions, guest quotas."""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app import rate_limiter
from app.hashing import sha256_hex
from app.guest_usage import GuestUsageKind, enforce_guest_limit


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clear_rate_limit_state():
    rate_limiter._attempts.clear()
    yield
    rate_limiter._attempts.clear()


# Chat session isolation moved to tests/test_chat_session_postgres.py
# (Postgres-backed store; no longer in-memory dicts).


# ── Rate limiter key isolation ────────────────────────────────────────────────

def test_rate_limiter_same_ip_shares_bucket():
    """Documented behavior: IP-only keys group coworkers on one NAT."""
    key = "search:203.0.113.1"
    for _ in range(3):
        assert rate_limiter.is_rate_limited(key, max_requests=3, window_seconds=60) is False
    assert rate_limiter.is_rate_limited(key, max_requests=3, window_seconds=60) is True


def test_rate_limiter_different_ips_are_independent():
    for _ in range(5):
        assert rate_limiter.is_rate_limited("login:10.0.0.1", max_requests=5, window_seconds=60) is False
    assert rate_limiter.is_rate_limited("login:10.0.0.1", max_requests=5, window_seconds=60) is True
    assert rate_limiter.is_rate_limited("login:10.0.0.2", max_requests=5, window_seconds=60) is False


def test_rate_limiter_key_suffix_isolates_users_on_same_ip():
    """Authenticated endpoints can pass user id so NAT coworkers don't block each other."""
    ip = "198.51.100.50"
    for user_id in ("42", "99"):
        key = f"claim_guest:{ip}:{user_id}"
        for _ in range(20):
            assert rate_limiter.is_rate_limited(key, max_requests=20, window_seconds=60) is False
        assert rate_limiter.is_rate_limited(key, max_requests=20, window_seconds=60) is True

    other_user_key = f"claim_guest:{ip}:100"
    assert rate_limiter.is_rate_limited(other_user_key, max_requests=20, window_seconds=60) is False


def test_require_rate_limit_raises_429(monkeypatch):
    async def _no_db(key, window):
        return None
    monkeypatch.setattr('app.rate_limiter._db_hit_count', _no_db)
    request = MagicMock()
    request.client = MagicMock(host="203.0.113.5")
    request.headers = {}

    for _ in range(2):
        asyncio.run(rate_limiter.require_rate_limit(
            request, max_requests=2, window_seconds=60, key_prefix="test_ep",
        ))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(rate_limiter.require_rate_limit(
            request, max_requests=2, window_seconds=60, key_prefix="test_ep",
        ))
    assert exc.value.status_code == 429


def test_require_rate_limit_uses_cloudflare_connecting_ip(monkeypatch):
    async def _no_db(key, window):
        return None
    monkeypatch.setattr('app.rate_limiter._db_hit_count', _no_db)
    """Behind a trusted proxy, rate-limit keys must use CF-Connecting-IP, not the peer."""
    rate_limiter._attempts.clear()
    monkeypatch.delenv("TRUST_FORWARDED_IP_HEADERS", raising=False)
    from app import guest_identity
    guest_identity._trusted_proxy_networks.cache_clear()

    request = MagicMock()
    request.client = MagicMock(host="127.0.0.1")  # trusted proxy peer (Apache/CF tunnel)
    request.headers = {"cf-connecting-ip": "198.51.100.20"}

    for _ in range(2):
        asyncio.run(rate_limiter.require_rate_limit(
            request, max_requests=2, window_seconds=60, key_prefix="signup_otp",
        ))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(rate_limiter.require_rate_limit(
            request, max_requests=2, window_seconds=60, key_prefix="signup_otp",
        ))
    assert exc.value.status_code == 429

    # Different CF client IP on the same peer must not share the bucket.
    other = MagicMock()
    other.client = MagicMock(host="127.0.0.1")
    other.headers = {"cf-connecting-ip": "198.51.100.21"}
    asyncio.run(rate_limiter.require_rate_limit(
        other, max_requests=2, window_seconds=60, key_prefix="signup_otp",
    ))


# ── Guest quota: same IP, different devices ───────────────────────────────────

def test_same_network_without_fingerprint_stays_independent(monkeypatch):
    """Coworkers on one NAT without fingerprint each keep separate quota rows."""
    monkeypatch.setattr("app.guest_usage._guest_ip_total_limit", lambda: 0)
    monkeypatch.setattr("app.guest_usage._guest_subnet_limit", lambda: 0)
    monkeypatch.setattr("app.guest_usage._guest_gen_limit", lambda: 3)

    same_ip = "deadbeef" * 4
    same_composite = sha256_hex("shared-office-signals")
    session_a = SimpleNamespace(
        id=1,
        session_id=str(uuid.uuid4()),
        ip_hash=same_ip,
        fingerprint_hash=None,
        composite_hash=same_composite,
        generation_count=3,
        chat_turn_count=0,
        voice_turn_count=0,
    )
    session_b = SimpleNamespace(
        id=2,
        session_id=str(uuid.uuid4()),
        ip_hash=same_ip,
        fingerprint_hash=None,
        composite_hash=same_composite,
        generation_count=0,
        chat_turn_count=0,
        voice_turn_count=0,
    )

    async def fake_get_row(db, session_id, ip_hash, *, fingerprint_hash=None, **kwargs):
        if session_id == session_a.session_id:
            return session_a
        return session_b

    monkeypatch.setattr("app.guest_usage.get_guest_session_row", fake_get_row)

    with pytest.raises(HTTPException) as blocked:
        _run(
            enforce_guest_limit(
                AsyncMock(),
                session_a.session_id,
                same_ip,
                GuestUsageKind.GENERATION,
                composite_hash=same_composite,
            )
        )
    assert blocked.value.status_code == 429

    allowed = _run(
        enforce_guest_limit(
            AsyncMock(),
            session_b.session_id,
            same_ip,
            GuestUsageKind.GENERATION,
            composite_hash=same_composite,
        )
    )
    assert allowed is session_b


def test_two_guest_sessions_same_ip_independent_when_ip_cap_off(monkeypatch):
    monkeypatch.setattr("app.guest_usage._guest_ip_total_limit", lambda: 0)
    monkeypatch.setattr("app.guest_usage._guest_subnet_limit", lambda: 0)
    monkeypatch.setattr("app.guest_usage._guest_gen_limit", lambda: 3)

    same_ip = "deadbeef" * 4
    session_a = SimpleNamespace(
        id=1,
        session_id=str(uuid.uuid4()),
        ip_hash=same_ip,
        fingerprint_hash=sha256_hex("fp-a"),
        generation_count=3,
        chat_turn_count=0,
        voice_turn_count=0,
    )
    session_b = SimpleNamespace(
        id=2,
        session_id=str(uuid.uuid4()),
        ip_hash=same_ip,
        fingerprint_hash=sha256_hex("fp-b"),
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
                AsyncMock(),
                session_a.session_id,
                same_ip,
                GuestUsageKind.GENERATION,
                fingerprint_hash=session_a.fingerprint_hash,
            )
        )
    assert blocked.value.status_code == 429

    allowed = _run(
        enforce_guest_limit(
            AsyncMock(),
            session_b.session_id,
            same_ip,
            GuestUsageKind.GENERATION,
            fingerprint_hash=session_b.fingerprint_hash,
        )
    )
    assert allowed is session_b
