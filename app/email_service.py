"""Transactional email via Resend API.

Burst handling
--------------
Resend free tier allows ~10 requests/second (and 100 emails/day). Concurrent
``asyncio.to_thread`` callers used to stampede the API and surface 429s to users.

This module throttles outbound Resend calls to roughly ``RESEND_MAX_RPS``
(default 6/s, under the 10/s ceiling) with a small concurrency cap, and retries
429 responses with backoff (honouring ``Retry-After`` when present).

Daily quota (100/day on free tier) cannot be smoothed by a queue — once the
day's allotment is exhausted, sends fail after retries and callers return a
clear error. One failed send does not block other users' queued attempts.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "noreply@your-verified-domain.com")

# Stay under Resend's ~10 req/s burst limit. Daily 100/day quota is separate —
# throttling only smooths short bursts; it cannot invent more daily capacity.
RESEND_MAX_RPS = float(os.getenv("RESEND_MAX_RPS", "6"))
RESEND_MAX_CONCURRENCY = int(os.getenv("RESEND_MAX_CONCURRENCY", "2"))
RESEND_MAX_RETRIES = int(os.getenv("RESEND_MAX_RETRIES", "4"))
RESEND_RETRY_BASE_SECONDS = float(os.getenv("RESEND_RETRY_BASE_SECONDS", "0.5"))
RESEND_RETRY_MAX_SECONDS = float(os.getenv("RESEND_RETRY_MAX_SECONDS", "8.0"))

_PURPOSE_LABELS = {
    "signup": "verify your Ducon account",
    "password_reset": "reset your Ducon password",
}

# Thread-safe throttle: sync httpx runs inside asyncio.to_thread worker threads.
_rate_lock = threading.Lock()
_last_send_mono = 0.0
_inflight = threading.Semaphore(max(1, RESEND_MAX_CONCURRENCY))


def _purpose_label(purpose: str) -> str:
    return _PURPOSE_LABELS.get(purpose, "complete your request")


def _min_interval() -> float:
    rps = RESEND_MAX_RPS if RESEND_MAX_RPS > 0 else 6.0
    return 1.0 / rps


def _acquire_send_slot() -> None:
    """Block until concurrency + inter-request spacing allow another Resend call."""
    _inflight.acquire()
    try:
        global _last_send_mono
        interval = _min_interval()
        with _rate_lock:
            now = time.monotonic()
            wait = (_last_send_mono + interval) - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            _last_send_mono = now
    except Exception:
        _inflight.release()
        raise


def _release_send_slot() -> None:
    _inflight.release()


def _parse_retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if not raw:
        return None
    raw = raw.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _backoff_seconds(attempt: int, response: httpx.Response | None = None) -> float:
    if response is not None:
        retry_after = _parse_retry_after(response)
        if retry_after is not None:
            # Small jitter so concurrent waiters don't re-stampede together.
            return min(RESEND_RETRY_MAX_SECONDS, retry_after + random.uniform(0.0, 0.25))
    delay = RESEND_RETRY_BASE_SECONDS * (2 ** attempt)
    delay = min(RESEND_RETRY_MAX_SECONDS, delay)
    return delay + random.uniform(0.0, 0.2)


def _post_resend(payload: dict[str, Any], *, timeout: float, log_label: str) -> None:
    """POST to Resend with throttle + 429 retry. Raises RuntimeError on failure."""
    if not RESEND_API_KEY:
        raise RuntimeError("RESEND_API_KEY is not configured")

    last_status: int | None = None
    for attempt in range(RESEND_MAX_RETRIES + 1):
        _acquire_send_slot()
        try:
            response = httpx.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {RESEND_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            _release_send_slot()
            if attempt >= RESEND_MAX_RETRIES:
                logger.error("Resend request failed for %s after retries", log_label)
                raise RuntimeError("Failed to send verification email") from exc
            delay = _backoff_seconds(attempt)
            logger.warning(
                "Resend transport error for %s (attempt %s/%s); retrying in %.2fs",
                log_label,
                attempt + 1,
                RESEND_MAX_RETRIES + 1,
                delay,
            )
            time.sleep(delay)
            continue

        delay: float | None = None
        try:
            if response.status_code == 429:
                last_status = 429
                if attempt >= RESEND_MAX_RETRIES:
                    logger.error(
                        "Resend rate-limited for %s after %s attempts (HTTP 429). "
                        "Daily free-tier quota (100/day) cannot be queued away.",
                        log_label,
                        RESEND_MAX_RETRIES + 1,
                    )
                    raise RuntimeError("Failed to send verification email")
                delay = _backoff_seconds(attempt, response)
                logger.warning(
                    "Resend HTTP 429 for %s (attempt %s/%s); retrying in %.2fs",
                    log_label,
                    attempt + 1,
                    RESEND_MAX_RETRIES + 1,
                    delay,
                )
            else:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    last_status = exc.response.status_code
                    logger.error(
                        "Resend API error for %s: HTTP %s",
                        log_label,
                        last_status,
                    )
                    raise RuntimeError("Failed to send verification email") from exc
                return
        finally:
            _release_send_slot()

        # Sleep only after releasing the slot so other senders are not blocked.
        if delay is not None:
            time.sleep(delay)
            continue

    logger.error(
        "Resend exhausted retries for %s (last_status=%s)",
        log_label,
        last_status,
    )
    raise RuntimeError("Failed to send verification email")


def send_otp_email(to: str, otp: str, purpose: str) -> None:
    """Send a one-time code email. Never log the OTP value."""
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
    _post_resend(payload, timeout=15.0, log_label=f"otp:{to}:{purpose}")


def send_contact_email(
    to: str,
    subject: str,
    text: str,
    html: str,
    attachments: list[dict] | None = None,
) -> None:
    """Send a contact / consultation email via Resend."""
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
        _post_resend(payload, timeout=30.0, log_label=f"contact:{to}")
    except RuntimeError as exc:
        # Preserve contact-specific wording for callers / logs.
        if "RESEND_API_KEY" in str(exc):
            raise
        logger.error("Resend contact email failed for %s", to)
        raise RuntimeError("Failed to send contact email") from exc
