"""Resend email throttle + 429 retry behaviour."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import httpx
import pytest

from app import email_service


@pytest.fixture(autouse=True)
def _reset_throttle_state(monkeypatch):
    """Isolate throttle globals and keep retries fast in unit tests."""
    monkeypatch.setattr(email_service, "RESEND_API_KEY", "test-key")
    monkeypatch.setattr(email_service, "RESEND_FROM_EMAIL", "noreply@example.com")
    monkeypatch.setattr(email_service, "RESEND_MAX_RPS", 1000.0)  # spacing ~0 in most tests
    monkeypatch.setattr(email_service, "RESEND_MAX_CONCURRENCY", 2)
    monkeypatch.setattr(email_service, "RESEND_MAX_RETRIES", 3)
    monkeypatch.setattr(email_service, "RESEND_RETRY_BASE_SECONDS", 0.01)
    monkeypatch.setattr(email_service, "RESEND_RETRY_MAX_SECONDS", 0.05)
    monkeypatch.setattr(email_service, "_last_send_mono", 0.0)
    # Fresh semaphore so prior tests don't leave it exhausted.
    import threading

    monkeypatch.setattr(email_service, "_inflight", threading.Semaphore(2))
    monkeypatch.setattr(email_service, "_rate_lock", threading.Lock())
    yield


def _ok_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {}
    resp.raise_for_status = MagicMock()
    return resp


def _http_response(status: int, headers: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    if status >= 400:
        err = httpx.HTTPStatusError(
            f"HTTP {status}",
            request=MagicMock(),
            response=resp,
        )
        resp.raise_for_status = MagicMock(side_effect=err)
    else:
        resp.raise_for_status = MagicMock()
    return resp


def test_send_otp_email_succeeds_on_first_try(monkeypatch):
    posts = []

    def fake_post(*args, **kwargs):
        posts.append(kwargs)
        return _ok_response()

    monkeypatch.setattr(email_service.httpx, "post", fake_post)
    email_service.send_otp_email("user@example.com", "123456", "signup")
    assert len(posts) == 1
    assert posts[0]["json"]["to"] == ["user@example.com"]


def test_retries_on_429_then_succeeds(monkeypatch):
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            return _http_response(429, {"Retry-After": "0.01"})
        return _ok_response()

    monkeypatch.setattr(email_service.httpx, "post", fake_post)
    monkeypatch.setattr(email_service.time, "sleep", lambda s: sleeps.append(s))

    email_service.send_otp_email("user@example.com", "123456", "signup")
    assert calls["n"] == 3
    assert sleeps  # backed off at least once


def test_respects_retry_after_header(monkeypatch):
    sleeps: list[float] = []
    calls = {"n": 0}
    monkeypatch.setattr(email_service, "RESEND_RETRY_MAX_SECONDS", 10.0)

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _http_response(429, {"Retry-After": "1.5"})
        return _ok_response()

    monkeypatch.setattr(email_service.httpx, "post", fake_post)
    monkeypatch.setattr(email_service.time, "sleep", lambda s: sleeps.append(s))

    email_service.send_otp_email("user@example.com", "999999", "signup")
    assert calls["n"] == 2
    # Retry-After 1.5 plus small jitter (0–0.25).
    assert 1.5 <= sleeps[0] <= 1.75


def test_exhausted_429_retries_raise_clear_error(monkeypatch):
    monkeypatch.setattr(email_service, "RESEND_MAX_RETRIES", 2)

    def always_429(*args, **kwargs):
        return _http_response(429, {"Retry-After": "0.01"})

    monkeypatch.setattr(email_service.httpx, "post", always_429)
    monkeypatch.setattr(email_service.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="Failed to send verification email"):
        email_service.send_otp_email("user@example.com", "123456", "signup")


def test_non_429_http_error_does_not_retry(monkeypatch):
    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        return _http_response(500)

    monkeypatch.setattr(email_service.httpx, "post", fake_post)
    monkeypatch.setattr(email_service.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="Failed to send verification email"):
        email_service.send_otp_email("user@example.com", "123456", "signup")
    assert calls["n"] == 1


def test_throttle_spaces_concurrent_sends(monkeypatch):
    """Inter-request spacing keeps effective rate under RESEND_MAX_RPS."""
    monkeypatch.setattr(email_service, "RESEND_MAX_RPS", 5.0)  # 200ms min interval
    monkeypatch.setattr(email_service, "RESEND_MAX_CONCURRENCY", 1)
    import threading

    monkeypatch.setattr(email_service, "_inflight", threading.Semaphore(1))
    monkeypatch.setattr(email_service, "_last_send_mono", 0.0)

    timestamps: list[float] = []

    def fake_post(*args, **kwargs):
        timestamps.append(time.monotonic())
        return _ok_response()

    monkeypatch.setattr(email_service.httpx, "post", fake_post)

    email_service.send_otp_email("a@example.com", "111111", "signup")
    email_service.send_otp_email("b@example.com", "222222", "signup")
    email_service.send_otp_email("c@example.com", "333333", "signup")

    assert len(timestamps) == 3
    gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    # Allow small timing slack on Windows.
    assert all(g >= 0.15 for g in gaps), gaps


def test_one_failure_does_not_poison_next_send(monkeypatch):
    """After a failed send, the next caller can still succeed (no shared poison)."""
    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _http_response(500)
        return _ok_response()

    monkeypatch.setattr(email_service.httpx, "post", fake_post)

    with pytest.raises(RuntimeError):
        email_service.send_otp_email("fail@example.com", "111111", "signup")

    email_service.send_otp_email("ok@example.com", "222222", "signup")
    assert calls["n"] == 2


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr(email_service, "RESEND_API_KEY", "")
    with pytest.raises(RuntimeError, match="RESEND_API_KEY"):
        email_service.send_otp_email("user@example.com", "123456", "signup")
