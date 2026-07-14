"""Deploy regression tests for multi-step signup modal (frontend)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SIGNUP = ROOT.parent / "Ducon_Library" / "src" / "components" / "auth" / "SignupForm.jsx"
FRONTEND_AUTH_MODAL = ROOT.parent / "Ducon_Library" / "src" / "components" / "auth" / "AuthModal.jsx"
FRONTEND_VALIDATION = ROOT.parent / "Ducon_Library" / "src" / "utils" / "signupFormValidation.js"


def test_signup_form_uses_progressive_steps():
    """Signup modal must split entry into profile and account steps before OTP."""
    if not FRONTEND_SIGNUP.is_file():
        pytest.skip(f"Frontend repo not found at {FRONTEND_SIGNUP}")

    source = FRONTEND_SIGNUP.read_text(encoding="utf-8")
    assert "formStep" in source
    assert "'profile'" in source or '"profile"' in source
    assert "'account'" in source or '"account"' in source
    assert "'otp'" in source or '"otp"' in source
    assert "validateSignupProfileStep" in source
    assert "validateSignupAccountStep" in source
    assert "Step 1 of 2" in source or "Step {stepNum} of 2" in source


def test_signup_validation_module_exports_step_validators():
    if not FRONTEND_VALIDATION.is_file():
        pytest.skip(f"Frontend validation not found at {FRONTEND_VALIDATION}")

    source = FRONTEND_VALIDATION.read_text(encoding="utf-8")
    for name in (
        "validateSignupProfileStep",
        "validateSignupAccountStep",
        "validateSignupBeforeOtp",
        "SIGNUP_FORM_STEPS",
    ):
        assert name in source


def test_signup_form_profile_step_fields_only():
    """Step 1 should collect name, email, phone — not password on the same screen."""
    if not FRONTEND_SIGNUP.is_file():
        pytest.skip(f"Frontend repo not found at {FRONTEND_SIGNUP}")

    source = FRONTEND_SIGNUP.read_text(encoding="utf-8")
    profile_block = re.search(
        r"if \(formStep === 'profile'\).*?(?=if \(formStep === 'otp'\)|// formStep === 'account')",
        source,
        re.DOTALL,
    )
    assert profile_block, "Could not locate profile step block"
    block = profile_block.group(0)
    assert 'name="username"' in block
    assert 'name="email"' in block
    assert 'name="phoneNumber"' in block
    assert 'name="password"' not in block


def test_signup_form_account_step_has_password_and_consents():
    if not FRONTEND_SIGNUP.is_file():
        pytest.skip(f"Frontend repo not found at {FRONTEND_SIGNUP}")

    source = FRONTEND_SIGNUP.read_text(encoding="utf-8")
    account_block = re.search(
        r"// formStep === 'account'.*",
        source,
        re.DOTALL,
    )
    assert account_block, "Could not locate account step block"
    block = account_block.group(0)
    assert 'name="password"' in block
    assert 'name="confirmPassword"' in block
    assert "agreeTerms" in block
    assert "agreeAge" in block


def test_auth_modal_fits_viewport_with_scroll():
    """Auth modal must cap height and scroll on small screens."""
    if not FRONTEND_AUTH_MODAL.is_file():
        pytest.skip(f"Frontend repo not found at {FRONTEND_AUTH_MODAL}")

    source = FRONTEND_AUTH_MODAL.read_text(encoding="utf-8")
    assert "auth-modal-body" in source
    assert "AuthModal.css" in source
    css_path = FRONTEND_AUTH_MODAL.parent / "AuthModal.css"
    assert css_path.is_file(), "AuthModal.css required for mobile layout"
    css = css_path.read_text(encoding="utf-8")
    assert "max-height" in css
    assert "overflow-y: auto" in css
    assert "100dvh" in css or "90dvh" in css
