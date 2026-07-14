"""
app/signed_urls.py
──────────────────
HMAC-signed URL helpers for locally-served (non-R2) images.

`<img>` tags cannot send Authorization headers, so local-mode image endpoints
that would otherwise be enumerable by id are protected with a short signature
derived from a server secret. The signed URL is generated server-side and handed
to the client, which uses it verbatim — no client change required.
"""
from __future__ import annotations

import hashlib
import hmac
import os

from app.config import IS_PRODUCTION

_DEFAULT_SECRET = "changeme-use-env-var-in-production"
_SIGN_SECRET = (
    os.getenv("URL_SIGNING_SECRET")
    or os.getenv("JWT_SECRET_KEY", _DEFAULT_SECRET)
)
if IS_PRODUCTION and (not _SIGN_SECRET or _SIGN_SECRET == _DEFAULT_SECRET):
    raise RuntimeError(
        "URL_SIGNING_SECRET or JWT_SECRET_KEY must be set to a strong secret in "
        "production (the default value is not allowed for signed guest URLs)."
    )


def _sign(namespace: str, identifier: int | str) -> str:
    msg = f"{namespace}:{identifier}".encode()
    return hmac.new(_SIGN_SECRET.encode(), msg, hashlib.sha256).hexdigest()


def sign_guest_generation(generation_id: int | str) -> str:
    return _sign("guest-gen", generation_id)


def verify_guest_generation(generation_id: int | str, token: str) -> bool:
    return hmac.compare_digest(_sign("guest-gen", generation_id), token or "")
