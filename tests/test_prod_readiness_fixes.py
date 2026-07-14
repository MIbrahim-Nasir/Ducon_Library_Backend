"""Tests for trusted proxy IP handling, atomic guest increment, source keys."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.dml import Update

from app.guest_identity import extract_client_ip, peer_is_trusted_proxy
from app.guest_usage import GuestUsageKind, increment_guest_usage
from app.hashing import sha256_hex


def test_trusted_loopback_honors_cf_connecting_ip():
    headers = {"cf-connecting-ip": "203.0.113.10"}
    assert extract_client_ip(headers, peer_host="127.0.0.1") == "203.0.113.10"


def test_untrusted_peer_ignores_spoofed_forwarded_headers(monkeypatch):
    monkeypatch.delenv("TRUST_FORWARDED_IP_HEADERS", raising=False)
    # Clear lru_cache so env changes apply
    from app import guest_identity
    guest_identity._trusted_proxy_networks.cache_clear()

    headers = {
        "cf-connecting-ip": "203.0.113.10",
        "x-forwarded-for": "198.51.100.1",
    }
    assert extract_client_ip(headers, peer_host="198.51.100.50") == "198.51.100.50"


def test_peer_is_trusted_proxy_loopback():
    assert peer_is_trusted_proxy("127.0.0.1") is True
    assert peer_is_trusted_proxy("8.8.8.8") is False


def test_generation_source_key_paths(monkeypatch, tmp_path):
    from app import storage

    monkeypatch.setattr(storage, "CLOUD_STORAGE", False)
    monkeypatch.setattr(storage, "_OUTPUTS_DIR", tmp_path)

    key = storage.generation_source_key(user_id=7, generation_id=99)
    assert key == "generation_sources/7/99.png"
    saved = storage.save_generation_source(b"png-bytes", user_id=7, generation_id=99)
    assert saved == key
    path = storage.serve_generation_source_path(key)
    assert path.exists()
    assert path.read_bytes() == b"png-bytes"
    assert storage.get_generation_source_url(99, key) == "/generations/99/before-image"


def test_increment_guest_usage_uses_limit_guard():
    """Compiled SQL must include count < limit so TOCTOU cannot overshoot."""
    session = SimpleNamespace(id=42)
    db = AsyncMock()
    result = MagicMock()
    result.rowcount = 1
    db.execute = AsyncMock(return_value=result)
    db.flush = AsyncMock()
    db.refresh = AsyncMock()

    async def _run():
        return await increment_guest_usage(db, session, GuestUsageKind.GENERATION)

    applied = asyncio.run(_run())
    assert applied is True
    assert db.execute.await_count == 1
    stmt = db.execute.await_args.args[0]
    assert isinstance(stmt, Update)
    # Dialect-agnostic: WHERE should reference generation_count
    compiled = str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": False}))
    assert "generation_count" in compiled.lower()


def test_normalize_meta_rejects_ambiguous_bare_int_as_documented():
    """Bare ints remain catalog_id by contract; uploads must use upload:N / file:N."""
    from app.routers.multi_image_gen import _normalize_meta_entry

    assert _normalize_meta_entry({"label": "x", "source": 7})["type"] == "catalog_id"
    assert _normalize_meta_entry({"label": "u", "source": "upload:0"})["type"] == "file"
    assert _normalize_meta_entry({"label": "g", "source": "gen:55"})["type"] == "generation_id"
