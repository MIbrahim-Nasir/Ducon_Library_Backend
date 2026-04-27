"""
app/storage.py
──────────────
Unified storage interface for AI generation images.

Switch between Cloudflare R2 and local disk with one setting:

    CLOUD_STORAGE = True   →  images go to Cloudflare R2 (private bucket)
    CLOUD_STORAGE = False  →  images stay on local disk under outputs/

Set via .env:
    USE_CLOUD_STORAGE = true   or   USE_CLOUD_STORAGE = false

Public API
──────────
    save_generation(user_id, filename)  -> stored_key
        Call after Gemini writes outputs/{user_id}/{filename}.
        Cloud: uploads to R2, removes the local temp file.
        Local: file stays on disk unchanged.
        Returns a stable key that is saved to Generation.url in the DB.

    delete_generation(stored_key)
        Removes from R2 or local disk.

    get_generation_url(generation_id, stored_key)  -> str
        Returns the URL the frontend uses to display the image:
        Cloud: fresh presigned R2 GET URL (expires in PRESIGNED_URL_EXPIRY s)
        Local: /generations/{id}/image  (auth-gated FastAPI endpoint)

    serve_local_path(stored_key)  -> Path
        Local mode only — absolute Path for FileResponse.
"""

import os
import functools
from pathlib import Path

import boto3

# ── Toggle ────────────────────────────────────────────────────────────────────
# Flip this (or set USE_CLOUD_STORAGE in .env) to switch storage backends.
# True  → Cloudflare R2 private bucket
# False → local disk  (outputs/{user_id}/{filename})
CLOUD_STORAGE: bool = os.getenv("USE_CLOUD_STORAGE", "false").lower() in ("true", "1", "yes")

# ── R2 settings (only used when CLOUD_STORAGE = True) ─────────────────────────
_R2_ENDPOINT_URL      = os.getenv("R2_ENDPOINT_URL",      "")
_R2_ACCESS_KEY_ID     = os.getenv("R2_ACCESS_KEY_ID",     "")
_R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
_R2_PRIVATE_BUCKET    = os.getenv("R2_PRIVATE_BUCKET",    "ducon-private")
_PRESIGNED_URL_EXPIRY = int(os.getenv("PRESIGNED_URL_EXPIRY", "3600"))

# ── Local settings ────────────────────────────────────────────────────────────
_OUTPUTS_DIR = Path("outputs")


@functools.lru_cache(maxsize=1)
def _r2():
    """Lazily created, cached boto3 S3 client for Cloudflare R2."""
    return boto3.client(
        "s3",
        endpoint_url=_R2_ENDPOINT_URL,
        aws_access_key_id=_R2_ACCESS_KEY_ID,
        aws_secret_access_key=_R2_SECRET_ACCESS_KEY,
        region_name="auto",   # required by the SDK; ignored by R2
    )


def _local_path(key: str) -> Path:
    """'generations/{user_id}/{filename}'  →  outputs/{user_id}/{filename}"""
    _, user_id, filename = key.split("/", 2)
    return _OUTPUTS_DIR / user_id / filename


# ── Public API ─────────────────────────────────────────────────────────────────

def save_generation(user_id: int, filename: str) -> str:
    """
    After Gemini saves outputs/{user_id}/{filename}:
      Cloud → upload to R2, delete local temp file.
      Local → file stays on disk as-is.
    Returns the key to store in Generation.url.
    """
    key   = f"generations/{user_id}/{filename}"
    local = _OUTPUTS_DIR / str(user_id) / filename

    if CLOUD_STORAGE:
        with open(local, "rb") as f:
            _r2().put_object(
                Bucket=_R2_PRIVATE_BUCKET,
                Key=key,
                Body=f,
                ContentType="image/png",
            )
        try:
            local.unlink()
        except OSError:
            pass

    return key


def delete_generation(stored_key: str) -> None:
    """Delete the generation from R2 or local disk."""
    if CLOUD_STORAGE:
        _r2().delete_object(Bucket=_R2_PRIVATE_BUCKET, Key=stored_key)
    else:
        path = _local_path(stored_key)
        if path.exists():
            path.unlink()


def get_generation_url(generation_id: int, stored_key: str) -> str:
    """
    URL the frontend uses to display/download a generation image.
      Cloud: presigned R2 GET URL valid for PRESIGNED_URL_EXPIRY seconds.
      Local: /generations/{id}/image  (FastAPI auth-gated endpoint).
    """
    if CLOUD_STORAGE:
        return _r2().generate_presigned_url(
            "get_object",
            Params={"Bucket": _R2_PRIVATE_BUCKET, "Key": stored_key},
            ExpiresIn=_PRESIGNED_URL_EXPIRY,
        )
    return f"/generations/{generation_id}/image"


def serve_local_path(stored_key: str) -> Path:
    """Local mode only — returns the absolute Path for FastAPI FileResponse."""
    return _local_path(stored_key)


# ── Guest generation storage ───────────────────────────────────────────────────

def save_guest_generation(session_id: str, filename: str) -> str:
    """
    After Gemini saves outputs/guests/{session_id}/{filename}:
      Cloud → upload to R2 under guests/{session_id}/{filename}, delete local temp file.
      Local → file stays on disk as-is.
    Returns the key to store in GuestGeneration.url.
    """
    key   = f"guests/{session_id}/{filename}"
    local = _OUTPUTS_DIR / "guests" / session_id / filename

    if CLOUD_STORAGE:
        with open(local, "rb") as f:
            _r2().put_object(
                Bucket=_R2_PRIVATE_BUCKET,
                Key=key,
                Body=f,
                ContentType="image/png",
            )
        try:
            local.unlink()
        except OSError:
            pass

    return key


def get_guest_generation_url(generation_id: int, stored_key: str) -> str:
    """
    URL the frontend uses to display a guest generation.
      Cloud: presigned R2 GET URL valid for PRESIGNED_URL_EXPIRY seconds.
      Local: /guest/generations/{id}/image
    """
    if CLOUD_STORAGE:
        return _r2().generate_presigned_url(
            "get_object",
            Params={"Bucket": _R2_PRIVATE_BUCKET, "Key": stored_key},
            ExpiresIn=_PRESIGNED_URL_EXPIRY,
        )
    return f"/guest/generations/{generation_id}/image"


def move_guest_to_user(session_id: str, filename: str, user_id: int) -> str:
    """
    Moves a guest generation file into the authenticated user's permanent storage.
    R2: copy-then-delete (R2 has no native rename/move).
    Local: filesystem rename.
    Returns the new permanent key stored in Generation.url.
    """
    guest_key = f"guests/{session_id}/{filename}"
    user_key  = f"generations/{user_id}/{filename}"

    if CLOUD_STORAGE:
        _r2().copy_object(
            Bucket=_R2_PRIVATE_BUCKET,
            CopySource={"Bucket": _R2_PRIVATE_BUCKET, "Key": guest_key},
            Key=user_key,
        )
        _r2().delete_object(Bucket=_R2_PRIVATE_BUCKET, Key=guest_key)
    else:
        src = _OUTPUTS_DIR / "guests" / session_id / filename
        dst = _OUTPUTS_DIR / str(user_id) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            src.rename(dst)

    return user_key


def delete_guest_generation(stored_key: str) -> None:
    """Delete a guest generation from R2 or local disk."""
    if CLOUD_STORAGE:
        _r2().delete_object(Bucket=_R2_PRIVATE_BUCKET, Key=stored_key)
    else:
        parts = stored_key.split("/", 2)   # ["guests", session_id, filename]
        if len(parts) == 3:
            path = _OUTPUTS_DIR / "guests" / parts[1] / parts[2]
            if path.exists():
                path.unlink()
