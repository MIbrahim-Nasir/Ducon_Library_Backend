"""Transactional email via Resend API."""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "noreply@your-verified-domain.com")

_PURPOSE_LABELS = {
    "signup": "verify your Ducon account",
    "password_reset": "reset your Ducon password",
}


def _purpose_label(purpose: str) -> str:
    return _PURPOSE_LABELS.get(purpose, "complete your request")


def send_otp_email(to: str, otp: str, purpose: str) -> None:
    """Send a one-time code email. Never log the OTP value."""
    if not RESEND_API_KEY:
        raise RuntimeError("RESEND_API_KEY is not configured")

    action = _purpose_label(purpose)
    subject = "Your Ducon verification code"
    text = (
        f"Your Ducon verification code is: {otp}\n\n"
        f"Use this code to {action}. "
        "This code expires in a few minutes.\n\n"
        "If you did not request this, you can ignore this email."
    )
    html = f"""\
<!DOCTYPE html>
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #111; line-height: 1.5;">
  <p>Your Ducon verification code is:</p>
  <p style="font-size: 28px; font-weight: 700; letter-spacing: 0.2em; margin: 16px 0;">{otp}</p>
  <p>Use this code to {action}. This code expires in a few minutes.</p>
  <p style="color: #666; font-size: 14px;">If you did not request this, you can ignore this email.</p>
</body>
</html>"""

    payload = {
        "from": RESEND_FROM_EMAIL,
        "to": [to],
        "subject": subject,
        "text": text,
        "html": html,
    }

    try:
        response = httpx.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Resend API error for %s (purpose=%s): HTTP %s",
            to,
            purpose,
            exc.response.status_code,
        )
        raise RuntimeError("Failed to send verification email") from exc
    except httpx.HTTPError as exc:
        logger.error("Resend request failed for %s (purpose=%s)", to, purpose)
        raise RuntimeError("Failed to send verification email") from exc


def send_contact_email(
    to: str,
    subject: str,
    text: str,
    html: str,
    attachments: list[dict] | None = None,
) -> None:
    """Send a contact / consultation email via Resend."""
    if not RESEND_API_KEY:
        raise RuntimeError("RESEND_API_KEY is not configured")
    if not to:
        raise RuntimeError("Contact recipient email is not configured")

    payload: dict = {
        "from": RESEND_FROM_EMAIL,
        "to": [to],
        "subject": subject,
        "text": text,
        "html": html,
    }
    if attachments:
        payload["attachments"] = attachments

    try:
        response = httpx.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error("Resend contact email error to %s: HTTP %s", to, exc.response.status_code)
        raise RuntimeError("Failed to send contact email") from exc
    except httpx.HTTPError as exc:
        logger.error("Resend contact email request failed for %s", to)
        raise RuntimeError("Failed to send contact email") from exc
