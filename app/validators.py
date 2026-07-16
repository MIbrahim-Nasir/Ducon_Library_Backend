"""
app/validators.py
─────────────────
Reusable field validators for email and phone number inputs.

Email rules (controlled by EMAIL_DOMAIN_POLICY):
  - allowlist (default): domain must be in TRUSTED_EMAIL_DOMAINS
  - allow_all: any syntactically valid email (Pydantic EmailStr upstream)
  - block_disposable: reject known temp/spam domains via disposable-email-domains
    (community blocklist used by PyPI and others)

Phone number rules:
  - Must be a valid UAE phone number
  - Accepts formats: +971XXXXXXXXX, 00971XXXXXXXXX, 05XXXXXXXX, 5XXXXXXXX
  - Both mobile (05X) and landline (02, 03, 04, 06, 07, 09) numbers accepted
  - Stored in E.164 international format: +971XXXXXXXXX
"""

from __future__ import annotations

import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberType, number_type

from disposable_email_domains import blocklist as DISPOSABLE_EMAIL_BLOCKLIST

from app.admin.settings_store import cfg

# ── Trusted email provider domains (allowlist mode) ───────────────────────────
# Extend this set as needed. All entries are lowercase.
TRUSTED_EMAIL_DOMAINS: set[str] = {
    # Google
    "gmail.com", "googlemail.com",
    # Microsoft
    "outlook.com", "hotmail.com", "hotmail.co.uk", "hotmail.fr", "hotmail.de",
    "hotmail.es", "hotmail.it", "live.com", "live.co.uk", "live.fr",
    "live.de", "live.com.au", "msn.com", "windowslive.com",
    # Yahoo
    "yahoo.com", "yahoo.co.uk", "yahoo.fr", "yahoo.de", "yahoo.es",
    "yahoo.it", "yahoo.com.au", "yahoo.co.in", "yahoo.co.jp",
    "ymail.com", "rocketmail.com",
    # Apple
    "icloud.com", "me.com", "mac.com",
    # ProtonMail
    "proton.me", "protonmail.com", "pm.me",
    # Zoho
    "zoho.com",
    # AOL / Verizon
    "aol.com", "verizon.net",
    # Samsung
    "samsung.com",
    # UAE providers
    "emirates.net.ae", "eim.ae", "du.ae", "etisalat.ae",
    # Ducon
    "duconodl.com",
}

EMAIL_DOMAIN_POLICY_DEFAULT = "allowlist"
EMAIL_DOMAIN_POLICIES = frozenset({"allowlist", "allow_all", "block_disposable"})


def email_domain_policy() -> str:
    """
    Effective email domain policy.

    Values: allowlist | allow_all | block_disposable

    ``ALLOW_ALL_EMAIL_DOMAINS=true`` is a convenience toggle that forces
    ``allow_all`` (overrides EMAIL_DOMAIN_POLICY). Prefer setting
    EMAIL_DOMAIN_POLICY explicitly.
    """
    allow_all = cfg("ALLOW_ALL_EMAIL_DOMAINS", False)
    if allow_all is True or str(allow_all).strip().lower() in {"1", "true", "yes", "on"}:
        return "allow_all"

    raw = str(cfg("EMAIL_DOMAIN_POLICY", EMAIL_DOMAIN_POLICY_DEFAULT) or "").strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw in EMAIL_DOMAIN_POLICIES:
        return raw
    return EMAIL_DOMAIN_POLICY_DEFAULT


def is_disposable_email_domain(domain: str) -> bool:
    """True if domain (or a parent suffix) is on the disposable blocklist."""
    domain = (domain or "").strip().lower().rstrip(".")
    if not domain or "." not in domain:
        return False
    parts = domain.split(".")
    # Check domain and parent suffixes (mail.temp.com → temp.com)
    for i in range(len(parts) - 1):
        if ".".join(parts[i:]) in DISPOSABLE_EMAIL_BLOCKLIST:
            return True
    return False


def validate_email_domain(email: str) -> str:
    """
    Validates the email domain according to EMAIL_DOMAIN_POLICY.
    Returns the normalised lowercase email if valid.
    Raises ValueError with a human-readable message if not.
    """
    email = email.strip().lower()
    try:
        domain = email.split("@", 1)[1]
    except IndexError:
        raise ValueError("Invalid email address.")
    if not domain:
        raise ValueError("Invalid email address.")

    policy = email_domain_policy()

    if policy == "allow_all":
        return email

    if policy == "block_disposable":
        if is_disposable_email_domain(domain):
            raise ValueError(
                "Temporary or disposable email addresses are not accepted. "
                "Please use a permanent email address."
            )
        return email

    # allowlist (default)
    if domain not in TRUSTED_EMAIL_DOMAINS:
        raise ValueError(
            f"Email domain '{domain}' is not accepted. "
            "Please use an email from a known provider such as Gmail, Outlook, "
            "Yahoo, iCloud, or @duconodl.com."
        )
    return email


def validate_uae_phone(phone: str) -> str:
    """
    Validates and normalises a UAE phone number.
    Accepts:
      +971XXXXXXXXX   (international)
      00971XXXXXXXXX  (international with 00 prefix)
      05XXXXXXXX      (local mobile)
      5XXXXXXXX       (local mobile without leading 0)
      04XXXXXXX       (local landline, e.g. Dubai)

    Returns the number in E.164 format (+971XXXXXXXXX).
    Raises ValueError with a human-readable message if invalid.
    """
    if not phone:
        raise ValueError("Phone number cannot be empty.")

    raw = phone.strip()

    # Normalise 00971 → +971 so phonenumbers library parses it correctly
    if raw.startswith("00971"):
        raw = "+" + raw[2:]

    try:
        parsed = phonenumbers.parse(raw, "AE")  # AE = UAE default region
    except NumberParseException:
        raise ValueError(
            "Invalid phone number. Please enter a valid UAE number, "
            "e.g. +971501234567 or 0501234567."
        )

    if not phonenumbers.is_valid_number(parsed):
        raise ValueError(
            "Phone number is not a valid UAE number. "
            "Please enter a valid UAE mobile or landline number."
        )

    # Restrict to UAE numbers only (country code 971)
    if parsed.country_code != 971:
        raise ValueError("Only UAE phone numbers (+971) are accepted.")

    ntype = number_type(parsed)
    allowed_types = {
        PhoneNumberType.MOBILE,
        PhoneNumberType.FIXED_LINE,
        PhoneNumberType.FIXED_LINE_OR_MOBILE,
        PhoneNumberType.VOIP,
    }
    if ntype not in allowed_types:
        raise ValueError(
            "Phone number type is not accepted. "
            "Please enter a UAE mobile or landline number."
        )

    # Return in E.164 format: +971XXXXXXXXX
    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
