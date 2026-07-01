"""
app/hashing.py
──────────────
Shared one-way hashing helper. Used to store non-reversible identifiers
(IP addresses, session ids) in audit/usage tables instead of raw PII.
"""
from __future__ import annotations

import hashlib


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
