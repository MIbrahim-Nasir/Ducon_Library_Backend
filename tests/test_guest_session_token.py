"""Guest session cookie — sign, verify, resolve."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.guest_session_token import (
    GUEST_SESSION_COOKIE,
    require_guest_session_id,
    resolve_guest_session_id,
    sign_guest_session_token,
    verify_guest_session_token,
)


class FakeRequest:
    def __init__(
        self,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
    ):
        self.headers = headers or {}
        self.cookies = cookies or {}


def test_sign_and_verify_round_trip():
    session_id = str(uuid.uuid4())
    token = sign_guest_session_token(session_id)
    assert verify_guest_session_token(token) == session_id


def test_verify_rejects_tampered_signature():
    session_id = str(uuid.uuid4())
    token = sign_guest_session_token(session_id)
    bad = token[:-1] + ("a" if token[-1] != "a" else "b")
    assert verify_guest_session_token(bad) is None


def test_resolve_prefers_header_over_cookie():
    session_id = str(uuid.uuid4())
    other_id = str(uuid.uuid4())
    request = FakeRequest(
        headers={"x-guest-session-id": session_id},
        cookies={GUEST_SESSION_COOKIE: sign_guest_session_token(other_id)},
    )
    assert resolve_guest_session_id(request) == session_id


def test_resolve_falls_back_to_cookie():
    session_id = str(uuid.uuid4())
    request = FakeRequest(
        cookies={GUEST_SESSION_COOKIE: sign_guest_session_token(session_id)},
    )
    assert resolve_guest_session_id(request) == session_id


def test_require_guest_session_id_raises_when_missing():
    with pytest.raises(HTTPException) as exc:
        require_guest_session_id(FakeRequest())
    assert exc.value.status_code == 401


def test_require_guest_session_id_rejects_invalid_uuid():
    with pytest.raises(HTTPException) as exc:
        require_guest_session_id(FakeRequest(headers={"x-guest-session-id": "not-a-uuid"}))
    assert exc.value.status_code == 400
