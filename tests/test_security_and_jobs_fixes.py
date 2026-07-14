"""Security + rate-limit + signed-URL + stuck-job regression tests."""
from __future__ import annotations

import asyncio
import importlib
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException


def test_signed_urls_fail_closed_in_production(monkeypatch):
    monkeypatch.setenv("URL_SIGNING_SECRET", "")
    monkeypatch.setenv("JWT_SECRET_KEY", "changeme-use-env-var-in-production")
    monkeypatch.setenv("ENV", "production")
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    import app.config as config

    importlib.reload(config)
    assert config.IS_PRODUCTION is True

    with pytest.raises(RuntimeError, match="URL_SIGNING_SECRET|JWT_SECRET_KEY"):
        import app.signed_urls as signed_urls

        importlib.reload(signed_urls)

    # Restore for subsequent tests in this process
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-for-unit-tests-xyz")
    importlib.reload(config)
    import app.signed_urls as signed_urls

    importlib.reload(signed_urls)
    assert signed_urls.sign_guest_generation(1)


def test_rate_limiter_auth_suffix_uses_jwt_sub_not_header_bytes(monkeypatch):
    async def _no_db(key, window):
        return None
    monkeypatch.setattr('app.rate_limiter._db_hit_count', _no_db)
    from app import rate_limiter
    from app.auth import create_access_token

    rate_limiter._attempts.clear()
    monkeypatch.delenv("TRUST_FORWARDED_IP_HEADERS", raising=False)

    token_a = create_access_token(42)
    token_b = create_access_token(99)
    # JWT headers are identical for HS256 — old bug used auth[7:23]
    assert token_a.split(".")[0] == token_b.split(".")[0]

    req_a = MagicMock()
    req_a.client = MagicMock(host="203.0.113.10")
    req_a.headers = {"authorization": f"Bearer {token_a}"}

    req_b = MagicMock()
    req_b.client = MagicMock(host="203.0.113.10")
    req_b.headers = {"authorization": f"Bearer {token_b}"}

    for _ in range(3):
        asyncio.run(rate_limiter.require_rate_limit(
            req_a, max_requests=3, window_seconds=60, key_prefix="test_auth_key",
        ))
    with pytest.raises(HTTPException) as blocked:
        asyncio.run(rate_limiter.require_rate_limit(
            req_a, max_requests=3, window_seconds=60, key_prefix="test_auth_key",
        ))
    assert blocked.value.status_code == 429

    # Different user on same IP must not share the bucket
    asyncio.run(rate_limiter.require_rate_limit(
        req_b, max_requests=3, window_seconds=60, key_prefix="test_auth_key",
    ))
    rate_limiter._attempts.clear()


def test_cleanup_old_keys_removes_stale():
    from app import rate_limiter
    import time

    rate_limiter._attempts.clear()
    rate_limiter._attempts["old"] = [time.time() - 7200]
    rate_limiter._attempts["new"] = [time.time()]
    removed = rate_limiter.cleanup_old_keys(max_age_seconds=3600)
    assert removed >= 1
    assert "old" not in rate_limiter._attempts
    assert "new" in rate_limiter._attempts
    rate_limiter._attempts.clear()


def test_resolve_user_from_token_honors_memory_revocation():
    from app.auth import create_access_token, revoke_token, resolve_user_from_token, decode_token_payload

    token = create_access_token(7)
    payload = decode_token_payload(token)
    jti = payload["jti"]
    exp = datetime.now(timezone.utc).timestamp() + 3600
    revoke_token(jti, exp)

    db = AsyncMock()
    result = asyncio.run(resolve_user_from_token(token, db))
    assert result is None
    db.execute.assert_not_called()


def test_generation_job_final_upserts_when_create_missing():
    """If create never persisted, final must still write a terminal row."""
    from app import generation_jobs as gj

    class _FakeSession:
        store: dict = {}

        def __init__(self):
            self._pending = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def add(self, row):
            self._pending.append(row)

        async def get(self, model, pk):
            return _FakeSession.store.get(pk)

        async def commit(self):
            for r in self._pending:
                ns = SimpleNamespace(
                    id=r.id,
                    user_id=r.user_id,
                    guest_session_id=getattr(r, "guest_session_id", None),
                    status=r.status,
                    events=list(getattr(r, "events", None) or []),
                    final=getattr(r, "final", None),
                    error=getattr(r, "error", None),
                    request_id=getattr(r, "request_id", None),
                    cancel_requested=False,
                    created_at=None,
                    updated_at=None,
                )
                _FakeSession.store[r.id] = ns
                # Keep mutable link for later attribute writes
                for attr in ("status", "final", "error", "updated_at"):
                    setattr(r, attr, getattr(ns, attr, None))
                # Map the ORM object id to the namespace for subsequent updates
                # after add+commit of upsert path — update fields on ns when
                # the same Python object is mutated after get returns None then add.
            # Prefer storing the real row objects so later attribute writes persist.
            for r in self._pending:
                _FakeSession.store[r.id] = r
            self._pending.clear()

    _FakeSession.store.clear()
    original = gj.async_session_maker
    gj.async_session_maker = lambda: _FakeSession()
    try:
        job = gj.create_job(user_id=3)
        job.status = "failed"
        job.error = "boom"
        asyncio.run(gj._persist_job_final(job, error="boom"))
        assert job.id in _FakeSession.store
        row = _FakeSession.store[job.id]
        assert row.status == "failed"
        assert row.error == "boom"
    finally:
        gj.async_session_maker = original
        gj.JOBS.clear()
        _FakeSession.store.clear()
