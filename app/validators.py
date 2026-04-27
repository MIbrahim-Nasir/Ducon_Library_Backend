"""
app/validators.py
─────────────────
Reusable field validators for email and phone number inputs.

Email rules:
  - Must be a valid email format (handled upstream by Pydantic EmailStr)
  - Domain must belong to a known, trusted email provider
  - Unknown / disposable / corporate-only domains are rejected

Phone number rules:
  - Must be a valid UAE phone number
  - Accepts formats: +971XXXXXXXXX, 00971XXXXXXXXX, 05XXXXXXXX, 5XXXXXXXX
  - Both mobile (05X) and landline (02, 03, 04, 06, 07, 09) numbers accepted
  - Stored in E.164 international format: +971XXXXXXXXX
"""

import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberType, number_type

# ── Trusted email provider domains ────────────────────────────────────────────
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
}


def validate_email_domain(email: str) -> str:
    """
    Validates that the email domain belongs to a trusted provider.
    Returns the normalised lowercase email if valid.
    Raises ValueError with a human-readable message if not.
    """
    email = email.strip().lower()
    try:
        domain = email.split("@")[1]
    except IndexError:
        raise ValueError("Invalid email address.")

    if domain not in TRUSTED_EMAIL_DOMAINS:
        raise ValueError(
            f"Email domain '{domain}' is not accepted. "
            "Please use an email from a known provider such as Gmail, Outlook, Yahoo, or iCloud."
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
