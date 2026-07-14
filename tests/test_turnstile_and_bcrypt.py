"""Turnstile dev bypass and passlib/bcrypt compatibility."""

from __future__ import annotations

import asyncio
import importlib

from passlib.context import CryptContext


def test_verify_turnstile_skips_in_development(monkeypatch):
    from app.routers import guest as guest_module

    monkeypatch.setattr(guest_module, "IS_PRODUCTION", False)
    monkeypatch.setattr(guest_module, "ENFORCE_TURNSTILE", False)
    monkeypatch.setattr(guest_module, "TURNSTILE_SECRET", "prod-secret-configured")

    assert asyncio.run(guest_module.verify_turnstile("", "127.0.0.1")) is True
    assert asyncio.run(guest_module.verify_turnstile("invalid", "127.0.0.1")) is True


def test_verify_turnstile_enforces_in_development_when_flag_set(monkeypatch):
    from app.routers import guest as guest_module

    monkeypatch.setattr(guest_module, "IS_PRODUCTION", False)
    monkeypatch.setattr(guest_module, "ENFORCE_TURNSTILE", True)
    monkeypatch.setattr(guest_module, "TURNSTILE_SECRET", "prod-secret")

    assert asyncio.run(guest_module.verify_turnstile("", "127.0.0.1")) is False


def test_verify_turnstile_fails_closed_production_without_secret(monkeypatch):
    from app.routers import guest as guest_module

    monkeypatch.setattr(guest_module, "IS_PRODUCTION", True)
    monkeypatch.setattr(guest_module, "ENFORCE_TURNSTILE", False)
    monkeypatch.setattr(guest_module, "TURNSTILE_SECRET", "")

    assert asyncio.run(guest_module.verify_turnstile("", "127.0.0.1")) is False


def test_passlib_bcrypt_hashes_without_version_warning(caplog):
    """passlib 1.7.4 must work with pinned bcrypt (no __about__ AttributeError)."""
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    with caplog.at_level("WARNING"):
        hashed = pwd_context.hash("test-password-123")
        assert pwd_context.verify("test-password-123", hashed)
        assert not any(
            "__about__" in rec.message or "bcrypt version" in rec.message
            for rec in caplog.records
        )


def test_bcrypt_pinned_below_4_1():
    bcrypt = importlib.import_module("bcrypt")
    version = getattr(bcrypt, "__version__", None)
    if version:
        major, minor, *_ = (int(x) for x in version.split("."))
        assert (major, minor) < (4, 1)
