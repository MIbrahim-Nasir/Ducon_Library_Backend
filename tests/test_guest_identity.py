"""Guest identity signals — IP extraction, subnet keys, composite server hash."""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.guest_identity import (
    build_guest_request_identity,
    composite_server_hash,
    extract_client_ip,
    extract_cloudflare_signals,
    subnet_key,
)
from app.guest_usage import (
    GuestUsageKind,
    enforce_guest_limit,
    get_guest_session_row,
    normalise_fingerprint_hash,
)
from app.hashing import sha256_hex


VALID_FP = "a" * 64
STORED_FP = sha256_hex(VALID_FP)


def _run(coro):
    return asyncio.run(coro)


# ── Client IP extraction ──────────────────────────────────────────────────────

def test_extract_client_ip_prefers_cf_connecting_ip():
    headers = {
        "cf-connecting-ip": "203.0.113.10",
        "x-forwarded-for": "198.51.100.1, 203.0.113.10",
    }
    assert extract_client_ip(headers, peer_host="127.0.0.1") == "203.0.113.10"


def test_extract_client_ip_falls_back_to_peer_host():
    assert extract_client_ip({}, peer_host="10.0.0.5") == "10.0.0.5"


def test_extract_client_ip_uses_x_forwarded_for_first_hop():
    headers = {"x-forwarded-for": "198.51.100.1, 203.0.113.10"}
    # Forwarded headers only apply when the immediate peer is a trusted proxy.
    assert extract_client_ip(headers, peer_host="127.0.0.1") == "198.51.100.1"


def test_extract_client_ip_ignores_xff_from_untrusted_peer():
    headers = {"x-forwarded-for": "198.51.100.1"}
    assert extract_client_ip(headers, peer_host="203.0.113.9") == "203.0.113.9"


# ── Subnet key ────────────────────────────────────────────────────────────────

def test_subnet_key_ipv4_truncates_to_slash_24():
    assert subnet_key("203.0.113.155") == "ipv4:203.0.113.0/24"


def test_subnet_key_ipv6_truncates_to_slash_64():
    assert subnet_key("2001:db8:abcd:0012:0000:0000:0000:00ff") == "ipv6:2001:db8:abcd:12::/64"


def test_subnet_key_unknown_ip_returns_none():
    assert subnet_key("unknown") is None


# ── Cloudflare signals + composite hash ───────────────────────────────────────

def test_extract_cloudflare_signals_reads_cf_headers():
    headers = {
        "cf-connecting-ip": "203.0.113.10",
        "cf-ja4": "t13d1516h2_8daaf6152771_d8a2da8f2df0",
        "cf-ipasn": "15169",
        "cf-ipcountry": "US",
        "accept-language": "en-US,en;q=0.9",
    }
    signals = extract_cloudflare_signals(headers)
    assert signals.ja4_fingerprint == "t13d1516h2_8daaf6152771_d8a2da8f2df0"
    assert signals.asn == "15169"
    assert signals.country == "US"
    assert signals.accept_language == "en-US,en;q=0.9"
    assert signals.subnet_key == "ipv4:203.0.113.0/24"


def test_composite_server_hash_stable_for_same_signals():
    headers = {
        "cf-connecting-ip": "203.0.113.10",
        "cf-ja4": "ja4-test",
        "cf-ipasn": "15169",
        "accept-language": "en-US",
    }
    signals = extract_cloudflare_signals(headers)
    h1 = composite_server_hash(signals)
    h2 = composite_server_hash(signals)
    assert h1 == h2
    assert h1 is not None
    assert len(h1) == 64


def test_composite_server_hash_none_without_signals():
    signals = extract_cloudflare_signals({})
    assert composite_server_hash(signals) is None


def test_build_guest_request_identity_normalises_fingerprint():
    headers = {"cf-connecting-ip": "203.0.113.10", "accept-language": "en"}
    identity = build_guest_request_identity(
        headers,
        peer_host="127.0.0.1",
        raw_fingerprint=VALID_FP,
    )
    assert identity.fingerprint_hash == STORED_FP
    assert identity.ip_hash == sha256_hex("203.0.113.10")
    assert identity.composite_hash is not None
    assert identity.subnet_key == "ipv4:203.0.113.0/24"


def test_normalise_fingerprint_hash_rejects_invalid():
    assert normalise_fingerprint_hash("not-valid") is None
    assert normalise_fingerprint_hash(VALID_FP) == STORED_FP


# ── Composite hash stored for audit only (no quota merge) ─────────────────────

def test_composite_hash_does_not_merge_different_uuids():
    """Same network signals must not merge separate session UUIDs into one quota row."""
    composite = sha256_hex("ja4|asn|lang|subnet")
    uuid_a = str(uuid.uuid4())
    uuid_b = str(uuid.uuid4())
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

    result = _run(
        get_guest_session_row(
            db,
            uuid_b,
            "abc",
            composite_hash=composite,
            subnet_key="ipv4:203.0.113.0/24",
        )
    )
    assert result.session_id == uuid_b
    assert len(added) == 1
    assert added[0].session_id == uuid_b
    assert added[0].composite_hash == composite
    assert len(execute_calls) == 1  # UUID lookup only (no composite merge query)


def test_fingerprint_takes_priority_over_composite():
    """When both signals match different rows, fingerprint wins."""
    fp_session = SimpleNamespace(
        id=1,
        session_id="fp-session",
        ip_hash="abc",
        fingerprint_hash=STORED_FP,
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
            result.scalar_one_or_none = lambda: fp_session
        else:
            result.scalar_one_or_none = lambda: None  # identity bind update
        return result

    db = AsyncMock()
    db.execute = fake_execute
    db.flush = AsyncMock()

    result = _run(
        get_guest_session_row(
            db,
            str(uuid.uuid4()),
            "abc",
            fingerprint_hash=STORED_FP,
            composite_hash=sha256_hex("other"),
        )
    )
    assert result is fp_session
    assert len(execute_calls) == 2  # fingerprint lookup + optional identity bind


# ── Optional subnet cap (disabled by default) ─────────────────────────────────

@pytest.fixture
def mock_db():
    return AsyncMock()


def test_subnet_cap_blocks_when_enabled(monkeypatch, mock_db):
    monkeypatch.setattr("app.guest_usage._guest_ip_total_limit", lambda: 0)
    monkeypatch.setattr("app.guest_usage._guest_subnet_limit", lambda: 20)
    subnet = "ipv4:203.0.113.0/24"

    async def fake_subnet_total(db, key):
        assert key == subnet
        return 20

    monkeypatch.setattr("app.guest_usage._subnet_total_usage", fake_subnet_total)
    monkeypatch.setattr(
        "app.guest_usage.get_guest_session_row",
        AsyncMock(side_effect=AssertionError("should not load session when subnet cap blocks")),
    )

    with pytest.raises(HTTPException) as exc:
        _run(
            enforce_guest_limit(
                mock_db,
                str(uuid.uuid4()),
                sha256_hex("203.0.113.10"),
                GuestUsageKind.GENERATION,
                subnet_key=subnet,
            )
        )
    assert exc.value.status_code == 429


def test_subnet_cap_zero_never_blocks(monkeypatch, mock_db):
    monkeypatch.setattr("app.guest_usage._guest_subnet_limit", lambda: 0)
    monkeypatch.setattr("app.guest_usage._guest_ip_total_limit", lambda: 0)
    monkeypatch.setattr("app.guest_usage._guest_gen_limit", lambda: 3)

    session = SimpleNamespace(
        id=1,
        session_id=str(uuid.uuid4()),
        ip_hash="abc",
        fingerprint_hash=None,
        composite_hash=None,
        subnet_key="ipv4:203.0.113.0/24",
        ja4_fingerprint=None,
        asn=None,
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
            subnet_key="ipv4:203.0.113.0/24",
        )
    )
    assert result is session
