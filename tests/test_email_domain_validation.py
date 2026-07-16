"""Email domain policy: allowlist / allow_all / block_disposable."""

from __future__ import annotations

import pytest

from app import validators


@pytest.fixture
def policy_env(monkeypatch):
    """Force policy via cfg without needing a DB-loaded settings cache."""

    def set_policy(policy: str = "allowlist", allow_all: bool = False):
        def _cfg(key, default=None):
            if key == "ALLOW_ALL_EMAIL_DOMAINS":
                return allow_all
            if key == "EMAIL_DOMAIN_POLICY":
                return policy
            return default

        monkeypatch.setattr(validators, "cfg", _cfg)

    return set_policy


def test_allowlist_accepts_trusted_and_duconodl(policy_env):
    policy_env("allowlist")
    assert validators.validate_email_domain("User@Gmail.com") == "user@gmail.com"
    assert validators.validate_email_domain("a@duconodl.com") == "a@duconodl.com"


def test_allowlist_rejects_unknown_corporate(policy_env):
    policy_env("allowlist")
    with pytest.raises(ValueError, match="not accepted"):
        validators.validate_email_domain("person@acme-corp.example")


def test_allow_all_accepts_unknown_domain(policy_env):
    policy_env("allow_all")
    assert validators.validate_email_domain("person@acme-corp.example") == "person@acme-corp.example"


def test_allow_all_bool_toggle_overrides_policy(policy_env):
    policy_env("allowlist", allow_all=True)
    assert validators.email_domain_policy() == "allow_all"
    assert validators.validate_email_domain("x@random-company.io") == "x@random-company.io"


def test_block_disposable_allows_gmail(policy_env):
    policy_env("block_disposable")
    assert validators.validate_email_domain("user@gmail.com") == "user@gmail.com"
    assert validators.validate_email_domain("ops@duconodl.com") == "ops@duconodl.com"


def test_block_disposable_rejects_known_temp(policy_env):
    policy_env("block_disposable")
    # Widely listed disposable domain from disposable-email-domains blocklist
    with pytest.raises(ValueError, match="[Dd]isposable|[Tt]emporary"):
        validators.validate_email_domain("user@mailinator.com")


def test_is_disposable_checks_parent_suffix():
    # Subdomain of a blocked domain should also match
    assert validators.is_disposable_email_domain("mailinator.com") is True
    assert validators.is_disposable_email_domain("gmail.com") is False


def test_invalid_email_missing_domain(policy_env):
    policy_env("allow_all")
    with pytest.raises(ValueError, match="Invalid"):
        validators.validate_email_domain("not-an-email")


def test_schema_user_create_respects_policy(policy_env, monkeypatch):
    """Pydantic UserCreate field validator uses validate_email_domain."""
    from app.db.schema import UserCreate

    policy_env("allow_all")
    u = UserCreate(
        email="corp@weird-domain.xyz",
        password="password123",
        name="Test",
    )
    assert u.email == "corp@weird-domain.xyz"

    policy_env("allowlist")
    with pytest.raises(Exception):
        UserCreate(
            email="corp@weird-domain.xyz",
            password="password123",
            name="Test",
        )
