"""Email OTP generation, persistence, and verification."""

from __future__ import annotations

import asyncio
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import hash_password, verify_password
from app.db.models import EmailOtp
from app.email_service import send_otp_email

OTP_EXPIRY_MINUTES = int(os.getenv("OTP_EXPIRY_MINUTES", "10"))
OTP_LENGTH = int(os.getenv("OTP_LENGTH", "6"))
OTP_MAX_ATTEMPTS = int(os.getenv("OTP_MAX_ATTEMPTS", "5"))
OTP_RESEND_COOLDOWN_SECONDS = int(os.getenv("OTP_RESEND_COOLDOWN_SECONDS", "60"))

PURPOSE_SIGNUP = "signup"
PURPOSE_PASSWORD_RESET = "password_reset"


def normalize_email(email: str) -> str:
    return email.strip().lower()


def generate_otp_code() -> str:
    upper = 10 ** OTP_LENGTH
    return str(secrets.randbelow(upper)).zfill(OTP_LENGTH)


def hash_otp(otp: str) -> str:
    return hash_password(otp)


def verify_otp_code(plain: str, otp_hash: str) -> bool:
    return verify_password(plain, otp_hash)


async def _get_latest_otp(
    db: AsyncSession,
    *,
    email: str,
    purpose: str,
) -> Optional[EmailOtp]:
    result = await db.execute(
        select(EmailOtp)
        .where(
            EmailOtp.email == email,
            EmailOtp.purpose == purpose,
            EmailOtp.verified_at.is_(None),
        )
        .order_by(EmailOtp.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def issue_otp(
    db: AsyncSession,
    *,
    email: str,
    purpose: str,
    pending_data: Optional[dict[str, Any]] = None,
) -> None:
    """Replace any pending OTP for this email/purpose and send a fresh code."""
    email = normalize_email(email)
    now = datetime.now(timezone.utc)

    existing = await _get_latest_otp(db, email=email, purpose=purpose)
    if existing is not None:
        elapsed = (now - existing.created_at).total_seconds()
        if elapsed < OTP_RESEND_COOLDOWN_SECONDS:
            wait = int(OTP_RESEND_COOLDOWN_SECONDS - elapsed)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Please wait {wait} seconds before requesting a new code.",
            )

    otp = generate_otp_code()
    expires_at = now + timedelta(minutes=OTP_EXPIRY_MINUTES)

    await db.execute(
        delete(EmailOtp).where(
            EmailOtp.email == email,
            EmailOtp.purpose == purpose,
            EmailOtp.verified_at.is_(None),
        )
    )

    record = EmailOtp(
        email=email,
        otp_hash=hash_otp(otp),
        purpose=purpose,
        expires_at=expires_at,
        attempts=0,
        pending_data=pending_data,
    )
    db.add(record)
    await db.commit()

    # Resend HTTP call is blocking — offload so the async route doesn't stall
    # the event loop (and other in-flight requests) while waiting on the API.
    # email_service throttles + retries 429s; daily Resend quota cannot be queued.
    try:
        await asyncio.to_thread(send_otp_email, email, otp, purpose)
    except Exception:
        # Drop the unused OTP so the per-email cooldown does not block a retry
        # after a transient Resend failure. Other users' OTPs are untouched.
        await db.execute(
            delete(EmailOtp).where(
                EmailOtp.email == email,
                EmailOtp.purpose == purpose,
                EmailOtp.verified_at.is_(None),
            )
        )
        await db.commit()
        raise


async def verify_otp(
    db: AsyncSession,
    *,
    email: str,
    otp: str,
    purpose: str,
    mark_verified: bool = True,
) -> EmailOtp:
    email = normalize_email(email)
    now = datetime.now(timezone.utc)

    record = await _get_latest_otp(db, email=email, purpose=purpose)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code.",
        )

    if record.expires_at < now:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code.",
        )

    if record.attempts >= OTP_MAX_ATTEMPTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many failed attempts. Please request a new code.",
        )

    if not verify_otp_code(otp.strip(), record.otp_hash):
        record.attempts += 1
        db.add(record)
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code.",
        )

    if mark_verified:
        record.verified_at = now
        db.add(record)
        await db.commit()
        await db.refresh(record)

    return record
