"""
POST /generate-multi-image
==========================
Endpoint called by the frontend when the chat agent requests a
`generate_multi_image` tool call.

Request (multipart/form-data)
──────────────────────────────
  prompt        str          Full task prompt (required)
  model         str          "pro" | "flash"  (default "pro")
  aspect_ratio  str          "1:1" | "4:3" | "3:4" | "16:9" | "9:16"  (optional)
  images_meta   JSON string  Ordered array describing each image:
  image_specs   JSON string  Optional compatibility alias for images_meta
                             [
                               {"label": "user space",     "source": "file:0"},
                               {"label": "Ducon pergola",  "source": 42},
                               {"label": "mood board",     "source": "wood_deck"},
                               {"label": "prev design",    "source": "gen:99"},
                               {"label": "inspiration",    "source": "https://..."},
                             ]
  files         List[UploadFile]  Repeated "files" fields. file:N references files[N].

Response (JSON)
───────────────
  {
    "id":              int,
    "generation_name": str,
    "url":             str,
    "signed_url":      str,
    "model_used":      str,
    "images_used":     [str, …]
  }

Frontend integration (from aiService.js)
─────────────────────────────────────────
  const form = new FormData();
  form.append("prompt", prompt);
  form.append("model",  model);                       // optional
  if (aspectRatio) form.append("aspect_ratio", aspectRatio);

  const imagesMeta = [];
  for (const img of images) {
    if (img.source instanceof File || img.source instanceof Blob) {
      imagesMeta.push({ label: img.label, type: "file" });
      form.append("files", img.source, img.label + ".jpg");
    } else if (/^\\d+$/.test(String(img.source))) {
      imagesMeta.push({ label: img.label, type: "catalog_id", source: String(img.source) });
    } else if (String(img.source).startsWith("gen:")) {
      imagesMeta.push({ label: img.label, type: "generation_id", source: img.source.slice(4) });
    } else if (String(img.source).startsWith("http")) {
      imagesMeta.push({ label: img.label, type: "url", source: img.source });
    } else {
      imagesMeta.push({ label: img.label, type: "catalog_name", source: img.source });
    }
  }
  form.append("images_meta", JSON.stringify(imagesMeta));

  const res = await fetch("/generate-multi-image", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });
  return await res.json();

Agent source convention (chat tool call → frontend resolution)
───────────────────────────────────────────────────────────────
  source 42 / "42"    → catalog_id
  source "gen:123"    → generation_id
  source "upload:0"   → first uploaded file in repeated "files" fields
  source "file:0"     → first uploaded file in repeated "files" fields
  source "https://…"  → URL             → type: url
  source "wood_deck"  → name string     → type: catalog_name
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncIterator, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_optional_user
from app.db.database import get_db, async_session_maker
from app.db.models import GuestSession, User
from app.image_utils import normalize_user_image
from app.routers.guest import resolve_guest_context
from app.sse import SSE_HEADERS
from app.error_logger import log_error, log_warning
from app.tool_generate_image import (
    MAX_IMAGES,
    ImageDescriptor,
    generate_multi_image,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate-multi-image", tags=["multi-image-gen"])

_VALID_TYPES    = {"catalog_id", "catalog_name", "generation_id", "url", "file"}
_VALID_MODELS   = {"pro", "flash"}
_VALID_RATIOS   = {"1:1", "4:3", "3:4", "16:9", "9:16"}

_KEEPALIVE_INTERVAL = 10.0  # seconds — well under Cloudflare's 100 s idle timeout


async def _stream_generation(
    descriptors,
    prompt,
    model,
    aspect_ratio,
    *,
    user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None,
    guest_session: Optional[GuestSession] = None,
    db,
) -> AsyncIterator[str]:
    """
    SSE generator that runs generate_multi_image in a background task and
    emits `: keepalive` comments every 10 s while waiting, so Cloudflare
    (Tunnel or CDN) never sees an idle connection and drops it at 100 s.

    Events:
      data: {"type": "status",  "message": "..."}   — informational, safe to ignore
      data: {"type": "done",    "id": ..., ...}      — success; same shape as old JSON response
      data: {"type": "error",   "message": "..."}    — failure; client should throw
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def worker():
        # Use a dedicated session — the request's `db` (from Depends(get_db))
        # is still bound to the SSE response and must not be used concurrently
        # from a background task (SQLAlchemy AsyncSession is not concurrency-safe).
        try:
            async with async_session_maker() as worker_db:
                worker_guest_session = None
                if guest_session_id:
                    result = await worker_db.execute(
                        select(GuestSession).where(GuestSession.session_id == guest_session_id)
                    )
                    worker_guest_session = result.scalar_one_or_none()
                    if worker_guest_session is None:
                        raise RuntimeError("Guest session was not found before saving generation.")

                # Chat/voice/studio quick-gen: always use the ImageGenAgent
                # prompt writer + evaluate + retry loop (enable_verify=True).
                result = await generate_multi_image(
                    user_id=user_id,
                    guest_session_id=guest_session_id,
                    guest_session=worker_guest_session,
                    prompt=prompt,
                    descriptors=descriptors,
                    model=model,
                    aspect_ratio=aspect_ratio or None,
                    db=worker_db,
                    enable_verify=True,
                )
                await worker_db.commit()
            await queue.put(("done", result))
        except (ValueError, RuntimeError) as exc:
            await log_error(
                "multi_image",
                "multi_image_gen.worker",
                str(exc),
                user_id=user_id,
                guest_session_id=guest_session_id,
                endpoint="/generate-multi-image",
                exc=exc,
            )
            await queue.put(("error", str(exc)))
        except Exception as exc:
            logger.exception("[MultiImageGen] Unexpected error in generation worker")
            await log_error(
                "multi_image",
                "multi_image_gen.worker",
                "Unexpected multi-image generation error",
                user_id=user_id,
                guest_session_id=guest_session_id,
                endpoint="/generate-multi-image",
                exc=exc,
            )
            await queue.put(("error", f"Unexpected error: {exc}"))

    task = asyncio.create_task(worker())
    try:
        # Emit an initial status so the client knows work started
        yield f"data: {json.dumps({'type': 'status', 'message': 'Generation started'})}\n\n"

        while True:
            try:
                event_type, payload = await asyncio.wait_for(
                    queue.get(), timeout=_KEEPALIVE_INTERVAL
                )
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            if event_type == "done":
                yield f"data: {json.dumps({'type': 'done', **payload})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': payload})}\n\n"
            break
    finally:
        await task


@router.post("")
@router.post("/")
async def create_multi_image_generation(
    request: Request,
    prompt: str = Form(..., description="Full task prompt for the generation model."),
    images_meta: Optional[str] = Form(
        None,
        description=(
            'JSON array of image descriptors. Each entry: {"label": str, "source": value}. '
            'source can be catalog id/name, "gen:123", "upload:0", "file:0", or URL. '
            'Legacy explicit type entries are also accepted.'
        ),
    ),
    image_specs: Optional[str] = Form(
        None,
        description="Compatibility alias for images_meta.",
    ),
    model: str = Form(
        "pro",
        description='"pro" (gemini-3-pro-image-preview) or "flash" (gemini-3.1-flash-image-preview)',
    ),
    aspect_ratio: Optional[str] = Form(
        None,
        description='Output aspect ratio: "1:1", "4:3", "3:4", "16:9", or "9:16".',
    ),
    files: List[UploadFile] = File(
        default=[],
        description='Repeated "files" fields. images_meta source "file:N" references files[N].',
    ),
    cf_turnstile_token: Optional[str] = Form(None),
    current_user: User | None = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a new image by compositing multiple source images according to a prompt.
    Uses Gemini image generation with the interleaved-label technique.
    """
    guest_session_id: str | None = None
    guest_session_row: GuestSession | None = None
    if current_user is None:
        guest_session_row = await resolve_guest_context(
            request, db,
            turnstile_token=cf_turnstile_token,
            endpoint="/generate-multi-image",
            source="multi_image_gen",
        )
        # Fingerprint remapping may return a different canonical UUID than the
        # header/cookie. Always persist and reload under the row's session_id
        # so the SSE worker's FK lookup cannot miss.
        guest_session_id = guest_session_row.session_id

    user_id = current_user.id if current_user else None

    descriptors = await parse_multi_image_request(
        prompt=prompt,
        model=model,
        aspect_ratio=aspect_ratio,
        images_meta=images_meta,
        image_specs=image_specs,
        files=files,
        log_tag="MultiImageGen",
    )

    # ── Generate (SSE so keepalive pings prevent Cloudflare proxy timeouts) ───
    if guest_session_row is not None:
        # The generation worker uses its own AsyncSession; commit the guest row
        # first so the worker can satisfy guest_generations' FK constraint.
        await db.commit()

    return StreamingResponse(
        _stream_generation(
            descriptors=descriptors,
            prompt=prompt,
            model=model,
            aspect_ratio=aspect_ratio,
            user_id=user_id,
            guest_session_id=guest_session_id,
            guest_session=guest_session_row,
            db=db,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


async def parse_multi_image_request(
    *,
    prompt: str,
    model: str,
    aspect_ratio: Optional[str],
    images_meta: Optional[str],
    image_specs: Optional[str],
    files: List[UploadFile],
    log_tag: str = "MultiImageGen",
) -> list[ImageDescriptor]:
    """Shared request validation + descriptor building for the multi-image
    endpoints (/generate-multi-image and /generation/jobs). One implementation
    so the two entry points cannot drift. Raises HTTPException(422) on any
    invalid input; returns the ordered ImageDescriptor list."""
    if not prompt.strip():
        raise HTTPException(status_code=422, detail="prompt must not be empty.")
    if model not in _VALID_MODELS:
        raise HTTPException(status_code=422, detail=f"model must be one of: {_VALID_MODELS}.")
    if aspect_ratio and aspect_ratio not in _VALID_RATIOS:
        raise HTTPException(status_code=422, detail=f"aspect_ratio must be one of: {_VALID_RATIOS}.")

    images_payload = images_meta or image_specs
    if not images_payload:
        raise HTTPException(status_code=422, detail="images_meta is required.")

    try:
        raw_meta: list[dict] = json.loads(images_payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"images_meta is not valid JSON: {exc}")

    if not isinstance(raw_meta, list) or len(raw_meta) == 0:
        raise HTTPException(status_code=422, detail="images_meta must be a non-empty JSON array.")
    if len(raw_meta) > MAX_IMAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Too many images: {len(raw_meta)} provided, maximum is {MAX_IMAGES}.",
        )
    if len(raw_meta) < 2:
        logger.warning(
            "[%s] Only %d image(s) — ImageGenAgent QC loop needs ≥2 "
            "(user space + Ducon design direction). Proceeding with best-effort single-pass.",
            log_tag, len(raw_meta),
        )

    normalized_meta: list[dict] = []
    for i, entry in enumerate(raw_meta):
        if not isinstance(entry, dict):
            raise HTTPException(status_code=422, detail=f"images_meta[{i}] must be an object.")
        if not entry.get("label"):
            raise HTTPException(status_code=422, detail=f"images_meta[{i}].label is required.")
        try:
            normalized = _normalize_meta_entry(entry)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"images_meta[{i}]: {exc}")
        t = normalized["type"]
        if t not in _VALID_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"images_meta[{i}].type must be one of: {_VALID_TYPES}.",
            )
        if t != "file" and normalized.get("source") in (None, ""):
            raise HTTPException(
                status_code=422,
                detail=f"images_meta[{i}].source is required when type != 'file'.",
            )
        normalized_meta.append(normalized)

    # Pre-check explicit file indexes before decoding uploads.
    for entry in normalized_meta:
        if entry["type"] != "file":
            continue
        file_index = entry.get("file_index")
        if file_index is None:
            continue
        if file_index < 0 or file_index >= len(files):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"images_meta source 'file:{file_index}' references files[{file_index}], "
                    f"but only {len(files)} file(s) were uploaded."
                ),
            )

    # Read and normalise uploaded files (CPU-bound decode off the event loop).
    file_pils: list = []
    for upload in files:
        raw_bytes = await upload.read()
        if not raw_bytes:
            raise HTTPException(status_code=422, detail=f"Uploaded file '{upload.filename}' is empty.")
        file_pils.append(await asyncio.to_thread(normalize_user_image, raw_bytes))

    file_idx = 0
    descriptors: list[ImageDescriptor] = []
    for entry in normalized_meta:
        t = entry["type"]
        if t == "file":
            selected_file_idx = entry.get("file_index")
            if selected_file_idx is None:
                if file_idx >= len(file_pils):
                    raise HTTPException(
                        status_code=422,
                        detail="Not enough uploaded files for type='file' images_meta entries.",
                    )
                selected_file_idx = file_idx
                file_idx += 1
            descriptors.append(ImageDescriptor(
                label=entry["label"],
                type="file",
                pil_image=file_pils[selected_file_idx],
            ))
        else:
            descriptors.append(ImageDescriptor(
                label=entry["label"],
                type=t,
                source=entry["source"],
            ))
    return descriptors


def _normalize_meta_entry(entry: dict) -> dict:
    """
    Accept both the current frontend contract:
        {"label": "...", "source": "file:0" | 123 | "gen:55" | "https://..." | "name"}

    and the older explicit contract:
        {"label": "...", "type": "catalog_id", "source": "123"}
    """
    label = entry["label"]
    source = entry.get("source")
    explicit_type = entry.get("type")

    if explicit_type:
        normalized = {"label": label, "type": explicit_type, "source": source}
        if explicit_type == "file":
            normalized["file_index"] = _parse_file_index(source)
        return normalized

    if isinstance(source, dict):
        source_id = source.get("id")
        source_type = source.get("_type")
        if source_type == "catalog_image" and source_id is not None:
            return {"label": label, "type": "catalog_id", "source": str(source_id)}
        if source_type in ("ai_generation", "generation") and source_id is not None:
            return {"label": label, "type": "generation_id", "source": str(source_id)}
        raise ValueError(
            "object sources must be resolved by the frontend unless they are catalog/generation refs."
        )

    if isinstance(source, int):
        return {"label": label, "type": "catalog_id", "source": str(source)}

    if not isinstance(source, str) or not source:
        raise ValueError("source is required.")

    if source.startswith(("file:", "upload:")):
        file_index = _parse_file_index(source)
        return {"label": label, "type": "file", "source": source, "file_index": file_index}

    if source.startswith("gen:"):
        gen_id = source.removeprefix("gen:")
        if not gen_id:
            raise ValueError("generation source must be in the form 'gen:123'.")
        return {"label": label, "type": "generation_id", "source": gen_id}

    if source.startswith(("http://", "https://")):
        return {"label": label, "type": "url", "source": source}

    if source.isdigit():
        return {"label": label, "type": "catalog_id", "source": source}

    return {"label": label, "type": "catalog_name", "source": source}


def _parse_file_index(source: object) -> Optional[int]:
    if source in (None, ""):
        return None
    if not isinstance(source, str) or not source.startswith(("file:", "upload:")):
        raise ValueError("file sources must be omitted or use the form 'file:N' or 'upload:N'.")
    raw_index = source.split(":", 1)[1]
    if not raw_index.isdigit():
        raise ValueError("file sources must use a numeric index, e.g. 'file:0' or 'upload:0'.")
    return int(raw_index)
