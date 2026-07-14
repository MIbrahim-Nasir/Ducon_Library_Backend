"""
app/storage.py
──────────────
Unified storage interface for AI generation images.

Switch between Cloudflare R2 and local disk with one setting:

    CLOUD_STORAGE = True   →  images go to Cloudflare R2 (private bucket)
    CLOUD_STORAGE = False  →  images stay on local disk under outputs/

Set via .env:
    USE_CLOUD_STORAGE = true   or   USE_CLOUD_STORAGE = false
    GENERATION_IMAGE_SERVE_MODE = redirect   or   proxy
        redirect (default) — GET /generations/{id}/image returns 302 to presigned R2
        proxy — stream image bytes through the API (same-origin; use if redirects fail)

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
import asyncio
import functools
import logging
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from PIL import Image

from app.error_logger import log_error

logger = logging.getLogger(__name__)

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

# Cloud image delivery: "proxy" (default — stream bytes through the API on the
# app's own origin, so images render regardless of R2 CORS config; same-origin
# like every major AI product) or "redirect" (302 to presigned R2 — bandwidth
# optimization, ONLY safe after R2_CORS_ORIGINS is applied via
# scripts/set_r2_cors.py, otherwise guest images blank on the CORS'd redirect).
GENERATION_IMAGE_SERVE_MODE: str = os.getenv(
    "GENERATION_IMAGE_SERVE_MODE", "proxy"
).strip().lower()

_IMAGE_RESPONSE_CACHE_CONTROL = "private, no-cache"


@functools.lru_cache(maxsize=1)
def _r2():
    """Lazily created, cached boto3 S3 client for Cloudflare R2."""
    # Retries + timeouts: a hung put_object used to lose paid generations with
    # no DB row and an orphaned local PNG.
    cfg = BotoConfig(
        retries={"max_attempts": 5, "mode": "standard"},
        connect_timeout=10,
        read_timeout=120,
        signature_version="s3v4",
    )
    return boto3.client(
        "s3",
        endpoint_url=_R2_ENDPOINT_URL,
        aws_access_key_id=_R2_ACCESS_KEY_ID,
        aws_secret_access_key=_R2_SECRET_ACCESS_KEY,
        region_name="auto",   # required by the SDK; ignored by R2
        config=cfg,
    )


def _local_path(key: str) -> Path:
    """Map a stored key to an absolute path under outputs/.

    Supported prefixes:
      generations/{user_id}/{filename}  →  outputs/{user_id}/{filename}
      guests/{session_id}/{filename}    →  outputs/guests/{session_id}/{filename}
    """
    parts = key.split("/", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid stored key: {key!r}")
    prefix, middle, filename = parts
    if prefix == "generations":
        result = (_OUTPUTS_DIR / middle / filename).resolve()
    elif prefix == "guests":
        result = (_OUTPUTS_DIR / "guests" / middle / filename).resolve()
    else:
        raise ValueError(f"Unsupported stored key prefix: {key!r}")
    try:
        result.relative_to(_OUTPUTS_DIR.resolve())
    except ValueError:
        raise ValueError(f"Refusing to serve path outside outputs directory: {key!r}")
    return result


def image_response_headers() -> dict[str, str]:
    """Cache headers for auth-gated generation image responses."""
    return {"Cache-Control": _IMAGE_RESPONSE_CACHE_CONTROL}


def should_proxy_generation_images() -> bool:
    """True when cloud images should be streamed through the API instead of 302."""
    return CLOUD_STORAGE and GENERATION_IMAGE_SERVE_MODE == "proxy"


# ── Public API ─────────────────────────────────────────────────────────────────

def _apply_watermark_to_local(local: Path) -> None:
    """Watermark a generated image file in place before it's uploaded/served.

    This is the SINGLE insertion point for the watermark: every generation flow
    (chat / studio / designer / guest / legacy two-image) writes its output to
    ``outputs/.../<filename>`` and then calls ``save_generation`` or
    ``save_guest_generation``. Watermarking here covers all of them exactly once
    — no double-watermarking, because each generation is saved exactly once and
    ``move_guest_to_user`` only renames/copies an already-watermarked file.

    Failures are logged and swallowed so a watermark rendering glitch can never
    block a generation from being saved.
    """
    try:
        from app.watermark import apply_watermark
        with Image.open(local) as im:
            im.load()
            marked = apply_watermark(im)
        marked.save(local, format="PNG")
    except Exception as exc:
        logger.warning("[storage] watermark step skipped for %s: %s", local, exc)


def save_generation(user_id: int, filename: str) -> str:
    """
    After Gemini saves outputs/{user_id}/{filename}:
      Cloud → upload to R2, delete local temp file.
      Local → file stays on disk as-is.
    Returns the key to store in Generation.url.
    """
    key   = f"generations/{user_id}/{filename}"
    local = _OUTPUTS_DIR / str(user_id) / filename

    _apply_watermark_to_local(local)

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


async def asave_generation(user_id: int, filename: str) -> str:
    """Async wrapper — offloads R2 upload + watermark PIL work to a thread
    so the event loop can keep sending SSE keepalives and serving other requests."""
    try:
        return await asyncio.to_thread(save_generation, user_id, filename)
    except Exception as exc:
        await log_error(
            "storage",
            "storage.asave_generation",
            f"Failed to persist generation {filename}",
            user_id=user_id,
            endpoint="storage.save_generation",
            exc=exc,
        )
        raise


async def asave_guest_generation(session_id: str, filename: str) -> str:
    """Async wrapper — see asave_generation."""
    try:
        return await asyncio.to_thread(save_guest_generation, session_id, filename)
    except Exception as exc:
        await log_error(
            "storage",
            "storage.asave_guest_generation",
            f"Failed to persist guest generation {filename}",
            guest_session_id=session_id,
            endpoint="storage.save_guest_generation",
            exc=exc,
        )
        raise


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


def read_generation_bytes(stored_key: str) -> bytes | None:
    """Read generation (or source) image bytes from R2 or local disk."""
    try:
        if CLOUD_STORAGE:
            obj = _r2().get_object(Bucket=_R2_PRIVATE_BUCKET, Key=stored_key)
            return obj["Body"].read()
        if stored_key.startswith(f"{GENERATION_SOURCE_PREFIX}/") or stored_key.startswith(
            f"{DESIGNER_INPUT_PREFIX}/"
        ):
            if stored_key.startswith(f"{GENERATION_SOURCE_PREFIX}/"):
                path = serve_generation_source_path(stored_key)
            else:
                path = serve_designer_input_path(stored_key)
        else:
            path = serve_local_path(stored_key)
        if not path.exists():
            return None
        return path.read_bytes()
    except Exception as exc:
        logger.warning("[storage] read_generation_bytes failed for %s: %s", stored_key, exc)
        return None


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

    _apply_watermark_to_local(local)

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
    from app.signed_urls import sign_guest_generation
    return f"/guest/generations/{generation_id}/image?token={sign_guest_generation(generation_id)}"


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


# ── Designer job input image storage ───────────────────────────────────────────
# The user's ORIGINAL space photo for a designer job — stored WITHOUT watermark
# (it is the client's own photo, not a Ducon generation). Keyed deterministically
# by (user_id, job_id) so the auth-gated serving endpoint can resolve it without a
# new DB column / migration: GET /designer/jobs/{job_id}/input-image derives the
# key from the authenticated user + path param.
DESIGNER_INPUT_PREFIX = "designer_inputs"


def designer_input_key(user_id: int, job_id: str) -> str:
    """Stable storage key for a designer job's user-space input image."""
    return f"{DESIGNER_INPUT_PREFIX}/{int(user_id)}/{job_id}.png"


def save_designer_input(user_id: int, job_id: str, image_bytes: bytes) -> str:
    """Persist the user's original space photo for a designer job (no watermark).

    Writes to local disk (outputs/designer_inputs/{user_id}/{job_id}.png) and, in
    cloud mode, uploads to R2 under the same key then removes the local temp file.
    Returns the stored key.
    """
    key = designer_input_key(user_id, job_id)
    local = _OUTPUTS_DIR / DESIGNER_INPUT_PREFIX / str(int(user_id)) / f"{job_id}.png"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(image_bytes)

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


async def asave_designer_input(user_id: int, job_id: str, image_bytes: bytes) -> str:
    """Async wrapper — offloads disk/R2 writes to a thread so the event loop can
    keep streaming designer job SSE events and serving other requests."""
    return await asyncio.to_thread(save_designer_input, user_id, job_id, image_bytes)


def get_designer_input_url(job_id: str, stored_key: str) -> str:
    """URL the frontend uses to display the designer job's input image.
      Cloud: presigned R2 GET URL valid for PRESIGNED_URL_EXPIRY seconds.
      Local: /designer/jobs/{job_id}/input-image  (FastAPI auth-gated endpoint).
    """
    if CLOUD_STORAGE:
        return _r2().generate_presigned_url(
            "get_object",
            Params={"Bucket": _R2_PRIVATE_BUCKET, "Key": stored_key},
            ExpiresIn=_PRESIGNED_URL_EXPIRY,
        )
    return f"/designer/jobs/{job_id}/input-image"


def serve_designer_input_path(stored_key: str) -> Path:
    """Local mode only — returns the absolute Path for FastAPI FileResponse."""
    parts = stored_key.split("/", 2)   # ["designer_inputs", user_id, "{job_id}.png"]
    if len(parts) != 3:
        raise ValueError(f"Invalid designer input key: {stored_key!r}")
    result = (_OUTPUTS_DIR / parts[0] / parts[1] / parts[2]).resolve()
    try:
        result.relative_to(_OUTPUTS_DIR.resolve())
    except ValueError:
        raise ValueError(f"Refusing to serve path outside outputs directory: {stored_key!r}")
    return result


def designer_input_exists(stored_key: str) -> bool:
    """Local mode only — True if the persisted input image is on disk."""
    try:
        return serve_designer_input_path(stored_key).exists()
    except ValueError:
        return False


# ── Generation source ("before") image ────────────────────────────────────────
GENERATION_SOURCE_PREFIX = "generation_sources"


def generation_source_key(*, user_id: int | None = None, guest_session_id: str | None = None, generation_id: int) -> str:
    """Stable storage key for a generation's user-space before photo."""
    if guest_session_id:
        return f"{GENERATION_SOURCE_PREFIX}/guests/{guest_session_id}/{int(generation_id)}.png"
    if user_id is None:
        raise ValueError("user_id or guest_session_id is required")
    return f"{GENERATION_SOURCE_PREFIX}/{int(user_id)}/{int(generation_id)}.png"


def save_generation_source(
    image_bytes: bytes,
    *,
    generation_id: int,
    user_id: int | None = None,
    guest_session_id: str | None = None,
) -> str:
    """Persist the user's original space photo for a generation (no watermark)."""
    key = generation_source_key(
        user_id=user_id,
        guest_session_id=guest_session_id,
        generation_id=generation_id,
    )
    if guest_session_id:
        local = (
            _OUTPUTS_DIR / GENERATION_SOURCE_PREFIX / "guests" / guest_session_id / f"{int(generation_id)}.png"
        )
    else:
        local = _OUTPUTS_DIR / GENERATION_SOURCE_PREFIX / str(int(user_id)) / f"{int(generation_id)}.png"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(image_bytes)

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


async def asave_generation_source(
    image_bytes: bytes,
    *,
    generation_id: int,
    user_id: int | None = None,
    guest_session_id: str | None = None,
) -> str:
    return await asyncio.to_thread(
        save_generation_source,
        image_bytes,
        generation_id=generation_id,
        user_id=user_id,
        guest_session_id=guest_session_id,
    )


def get_generation_source_url(generation_id: int, stored_key: str) -> str:
    """URL for the before photo.
      Cloud: fresh presigned R2 GET URL.
      Local / auth path: /generations/{id}/before-image
    """
    if CLOUD_STORAGE:
        return _r2().generate_presigned_url(
            "get_object",
            Params={"Bucket": _R2_PRIVATE_BUCKET, "Key": stored_key},
            ExpiresIn=_PRESIGNED_URL_EXPIRY,
        )
    return f"/generations/{generation_id}/before-image"


def serve_generation_source_path(stored_key: str) -> Path:
    parts = stored_key.split("/")
    if len(parts) < 3 or parts[0] != GENERATION_SOURCE_PREFIX:
        raise ValueError(f"Invalid generation source key: {stored_key!r}")
    result = (_OUTPUTS_DIR.joinpath(*parts)).resolve()
    try:
        result.relative_to(_OUTPUTS_DIR.resolve())
    except ValueError:
        raise ValueError(f"Refusing to serve path outside outputs directory: {stored_key!r}")
    return result
