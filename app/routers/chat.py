"""
Chat Agent Router
=================
Provides two SSE endpoints that together implement the full chat turn loop:

  POST /chat/message
    Send a user message (text + optional images).
    Returns an SSE stream with text deltas, thinking deltas, tool calls, and
    a final `done` event carrying the `interaction_id` for the next turn.

  POST /chat/tool_result
    Submit the results of tool calls the model requested.
    Returns an SSE stream for the model's follow-up response.
    Must be called with the `interaction_id` returned from /chat/message.

Frontend loop
─────────────
1.  POST /chat/message              → SSE stream
    ├─ data: {"type":"text_delta","text":"..."}    (stream to UI)
    ├─ data: {"type":"thinking_delta","text":"..."}(optional, show/hide)
    ├─ data: {"type":"tool_call","id":"...","name":"...","args":{...}} (execute)
    └─ data: {"type":"done","interaction_id":"v1_..."}

2a. If any tool_call events arrived:
    Execute each tool (window.__duconAPI.<name>(args))
    POST /chat/tool_result  { previous_interaction_id, results:[{call_id,name,result,error},...] }
    → SSE stream (same event types as step 1 — may trigger more tool calls)

2b. No tool calls → conversation turn complete; save `interaction_id` for next message.

Repeat from step 1 for the next user message, passing `previous_interaction_id`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user, get_optional_user
from app.db.database import get_db
from app.db.models import GuestSession, User
from app.guest_identity import build_guest_request_identity
from app.guest_session_token import require_guest_session_id
from app.guest_usage import GuestUsageKind, get_guest_session_row, increment_guest_usage
from app.routers.guest import resolve_guest_context
from app import chat_agent
from app import chat_session
from app import llm_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])
from app.admin.settings_store import cfg
from app.error_logger import log_error, log_warning

_MAX_TOOL_RESULT_CHARS = 12000      # default; live value via cfg("CHAT_TOOL_RESULT_MAX_CHARS", _MAX_TOOL_RESULT_CHARS)
_MAX_AISEARCH_ITEMS = 8
_MAX_UPLOAD_MB_DEFAULT = 50


def _max_tool_result_chars() -> int:
    return int(cfg("CHAT_TOOL_RESULT_MAX_CHARS", _MAX_TOOL_RESULT_CHARS))


def _max_aisearch_items() -> int:
    return int(cfg("CHAT_AISEARCH_MAX_ITEMS", _MAX_AISEARCH_ITEMS))


def _max_upload_bytes() -> int:
    return int(cfg("MAX_UPLOAD_SIZE_MB", _MAX_UPLOAD_MB_DEFAULT)) * 1024 * 1024

# Magic-byte signatures for allowed file types
_ALLOWED_MIME_PREFIXES = (
    "image/",
    "application/pdf",
)
_MAGIC_SIGNATURES: dict[bytes, str] = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",     # RIFF....WEBP — check further below
    b"\x00\x00\x00\x0cjP  ": "image/jp2",
    b"\x49\x49\x2a\x00": "image/tiff",
    b"\x4d\x4d\x00\x2a": "image/tiff",
    b"%PDF": "application/pdf",
    b"\x00\x00\x00": "video/mp4",   # blocked further in mime check
}
_BLOCKED_EXTENSIONS = {".exe", ".sh", ".bat", ".cmd", ".js", ".py", ".php", ".rb", ".ps1"}


def _sniff_mime(data: bytes) -> str | None:
    """Return a MIME type guessed from the first few bytes, or None if unknown."""
    header = data[:16]
    for sig, mime in _MAGIC_SIGNATURES.items():
        if header[:len(sig)] == sig:
            # Extra WebP discriminator
            if sig == b"RIFF" and data[8:12] != b"WEBP":
                return None
            return mime
    return None


def _validate_upload(filename: str, data: bytes) -> None:
    """Raise 400/413 if the upload violates size or MIME policy."""
    _max_up = _max_upload_bytes()
    if len(data) > _max_up:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {_max_up // (1024*1024)} MB.",
        )
    ext = os.path.splitext(filename.lower())[1]
    if ext in _BLOCKED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not allowed.")
    sniffed = _sniff_mime(data)
    if sniffed is not None:
        if not any(sniffed.startswith(p) for p in _ALLOWED_MIME_PREFIXES):
            raise HTTPException(status_code=400, detail="File type not allowed.")


# ── Request / response models ─────────────────────────────────────────────────

class ToolResultItem(BaseModel):
    call_id: str
    name:    str
    result:  object          # any JSON-serialisable value
    error:   bool = False


class ToolResultRequest(BaseModel):
    previous_interaction_id: str
    results: List[ToolResultItem]


# ── Shared SSE response config ────────────────────────────────────────────────

from app.sse import SSE_HEADERS as _SSE_HEADERS


# Cloudflare idle-kills ~100s; keep well under that. Must NOT wrap
# ``aiter.__anext__()`` in ``asyncio.wait_for`` — a timeout cancels the
# underlying Gemini/Claude stream and the client sees "SSE closed without done".
_CHAT_SSE_KEEPALIVE_INTERVAL = 15.0


async def _stream_with_session_save(
    user_id: int | None,
    generator,
    *,
    guest_session_id: str | None = None,
    guest_session_row: GuestSession | None = None,
    db: AsyncSession | None = None,
    user_text: str | None = None,
    record_transcript: bool = True,
    increment_guest_on_done: bool = False,
    keepalive_interval: float | None = None,
):
    """Emit chat SSE, then persist final-turn metadata without hiding ``done``.

    The ``done`` event is the client-visible transaction boundary.  It must be
    yielded before any best-effort persistence: a database/session failure must
    not truncate an otherwise completed model response and leave the composer
    permanently in its streaming state.

    Keepalives are produced via a background producer + ``queue.get`` timeout
    (same pattern as multi-image / studio SSE). Waiting on the async generator
    with ``wait_for`` would cancel in-flight model calls on every idle gap.
    """
    interval = (
        _CHAT_SSE_KEEPALIVE_INTERVAL
        if keepalive_interval is None
        else float(keepalive_interval)
    )
    assistant_parts: list[str] = []
    queue: asyncio.Queue = asyncio.Queue()

    async def _producer() -> None:
        try:
            async for chunk in generator:
                await queue.put(("chunk", chunk))
            await queue.put(("end", None))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Chat SSE producer failed")
            await queue.put(("error", exc))

    task = asyncio.create_task(_producer())
    try:
        # First byte immediately so Cloudflare/proxies do not treat the origin
        # as hung while Gemini uploads or TTFT are still in flight.
        yield ": keepalive\n\n"
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=interval)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            if kind == "end":
                break

            if kind == "error":
                yield chat_agent._sse_error(str(payload))
                break

            chunk = payload
            if chunk.startswith("data:"):
                try:
                    event = json.loads(chunk[5:].strip())
                    if event.get("type") == "text_delta":
                        delta = event.get("text") or ""
                        if delta:
                            assistant_parts.append(delta)
                    if event.get("type") == "done" and event.get("interaction_id"):
                        iid = event["interaction_id"]
                        # `done` is the protocol boundary and is always the final
                        # event from chat_agent. Deliver it before persistence so a
                        # database/session error cannot suppress completion.
                        yield chunk
                        try:
                            if user_id is not None:
                                await chat_session.set_interaction_id(user_id, iid)
                            elif guest_session_id:
                                await chat_session.set_guest_interaction_id(
                                    guest_session_id, iid
                                )
                            if record_transcript:
                                model_text = "".join(assistant_parts).strip()
                                memory_user = (user_text or "").strip()
                                if memory_user or model_text:
                                    if user_id is not None:
                                        await chat_session.append_turn(
                                            user_id, memory_user, model_text
                                        )
                                    elif guest_session_id:
                                        await chat_session.append_guest_turn(
                                            guest_session_id, memory_user, model_text
                                        )
                            if (
                                increment_guest_on_done
                                and guest_session_row is not None
                                and db is not None
                            ):
                                await increment_guest_usage(
                                    db, guest_session_row, GuestUsageKind.CHAT
                                )
                                await db.commit()
                        except Exception as exc:
                            logger.exception(
                                "Chat turn completed but final persistence failed "
                                "(user_id=%r guest_session_id=%r interaction_id=%r)",
                                user_id,
                                guest_session_id,
                                iid,
                            )
                            if db is not None:
                                try:
                                    await db.rollback()
                                except Exception:
                                    logger.exception(
                                        "Failed to roll back chat persistence transaction"
                                    )
                            await log_error(
                                "chat",
                                "chat._stream_with_session_save",
                                str(exc),
                                user_id=user_id,
                                guest_session_id=guest_session_id,
                                endpoint="/chat/message",
                                exc=exc,
                            )
                        continue
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            yield chunk
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Chat SSE producer cleanup failed")


# ── GET /chat/session ─────────────────────────────────────────────────────────

@router.get("/session")
async def get_chat_session(
    request: Request,
    current_user: User | None = Depends(get_optional_user),
):
    """Return the persisted Interactions API chain id (voice + text share it)."""
    if current_user is not None:
        return {"interaction_id": await chat_session.get_interaction_id(current_user.id)}
    x_guest_session_id = require_guest_session_id(request)
    return {
        "interaction_id": await chat_session.get_guest_interaction_id(x_guest_session_id),
    }


@router.delete("/session", status_code=204)
async def clear_chat_session(
    request: Request,
    current_user: User | None = Depends(get_optional_user),
):
    """Clear the stored chat session (e.g. when user clears the conversation)."""
    if current_user is not None:
        await chat_session.clear_session(current_user.id)
        return
    x_guest_session_id = require_guest_session_id(request)
    await chat_session.clear_guest_session(x_guest_session_id)


# ── POST /chat/message ────────────────────────────────────────────────────────

@router.post("/message")
async def chat_message(
    request: Request,
    message: Optional[str] = Form(
        None,
        description="User's text message. May be omitted when only files are sent.",
    ),
    previous_interaction_id: Optional[str] = Form(
        None,
        description=(
            "ID from the previous turn's `done` event. "
            "Omit or pass null to start a new conversation."
        ),
    ),
    cf_turnstile_token: Optional[str] = Form(None),
    files: List[UploadFile] = File(
        default=[],
        description="Images or documents to attach to this message.",
    ),
    current_user: User | None = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a user message and receive a streaming SSE response.

    Accepts multipart/form-data with an optional text message and optional
    file attachments (images, documents).  Attached files are uploaded to the
    Gemini Files API and included as multimodal input parts.

    Returns a streaming SSE response.  Each data event is a JSON object:
      {"type":"text_delta","text":"..."}
      {"type":"thinking_delta","text":"..."}
      {"type":"tool_call","id":"...","name":"...","args":{...}}
      {"type":"done","interaction_id":"..."}
      {"type":"error","message":"..."}
    """
    if not message and not files:
        raise HTTPException(status_code=422, detail="Provide at least a message or a file.")

    guest_session_id: str | None = None
    guest_session_row: GuestSession | None = None
    if current_user is None:
        guest_session_row = await resolve_guest_context(
            request, db,
            turnstile_token=cf_turnstile_token,
            endpoint="/chat/message",
            source="chat",
            usage_kind=GuestUsageKind.CHAT,
        )
        # Fingerprint remapping may return a different canonical UUID than the
        # header/cookie — bind chat usage/history to the row that was enforced.
        guest_session_id = guest_session_row.session_id

    chat_agent._dbg(
        "[CHAT ROUTER ▶ MESSAGE]",
        {
            "user_id": current_user.id if current_user else None,
            "guest_session_id": guest_session_id,
            "message_chars": len(message or ""),
            "message_preview": (message or "")[:500],
            "previous_interaction_id": previous_interaction_id,
            "files": [
                {
                    "filename": f.filename,
                    "content_type": f.content_type,
                }
                for f in files
            ],
        },
    )

    # ── Read + validate attachments locally (fast) ────────────────────────────
    # Gemini Files API upload is slow and used to run *before* StreamingResponse,
    # so Cloudflare/browsers often saw no first byte and raised
    # ERR_HTTP2_PROTOCOL_ERROR / Failed to fetch. Uploads run inside the SSE
    # producer below so keepalives can flow immediately.
    pending_files: list[tuple[str, bytes, str]] = []
    for upload in files:
        if not upload.filename:
            continue
        file_bytes = await upload.read()
        if not file_bytes:
            continue
        _validate_upload(upload.filename, file_bytes)
        mime = upload.content_type or _infer_mime(upload.filename)
        pending_files.append((upload.filename, file_bytes, mime))

    # Prefer server-persisted session (updated by voice_context injects) over a
    # possibly stale client previous_interaction_id.
    # After guest → login, the client clears previous_interaction_id so this
    # fallback cannot chain onto a guest turn.
    if current_user is not None:
        session_prev = await chat_session.get_interaction_id(current_user.id)
    else:
        session_prev = await chat_session.get_guest_interaction_id(guest_session_id)
    effective_prev = session_prev or previous_interaction_id
    memory_user = (message or "").strip()
    if not memory_user and pending_files:
        memory_user = "[User sent image attachment(s)]"
    stream_user_id = current_user.id if current_user else None
    use_claude = llm_provider.use_claude()

    async def _produce_message_stream():
        # Gemini path → upload to the Files API and reference by URI.
        # Claude path → Anthropic Messages API is stateless and cannot read Gemini
        #   File URIs, so we send the bytes inline as base64 image/document blocks.
        file_parts: list[dict] = []
        for filename, file_bytes, mime in pending_files:
            if use_claude:
                import base64 as _b64
                if mime.startswith("image/"):
                    claude_mime, claude_bytes = _normalize_image_for_claude(file_bytes, mime)
                    file_parts.append({
                        "type": "image_b64",
                        "data": _b64.b64encode(claude_bytes).decode("ascii"),
                        "mime_type": claude_mime,
                    })
                elif mime == "application/pdf":
                    file_parts.append({
                        "type": "document_b64",
                        "data": _b64.b64encode(file_bytes).decode("ascii"),
                        "mime_type": mime,
                    })
                else:
                    logger.info(
                        "Skipping non-image/pdf attachment for Claude: %s (%s)",
                        filename,
                        mime,
                    )
                chat_agent._dbg(
                    "[CHAT ROUTER ▶ FILE INLINE (claude)]",
                    {"filename": filename, "bytes": len(file_bytes), "mime_type": mime},
                )
                continue

            try:
                uploaded = await chat_agent.upload_file_to_gemini(
                    file_bytes=file_bytes,
                    filename=filename,
                    mime_type=mime,
                )
                chat_agent._dbg(
                    "[CHAT ROUTER ▶ FILE UPLOADED]",
                    {
                        "filename": filename,
                        "bytes": len(file_bytes),
                        "mime_type": mime,
                        "gemini_uri": uploaded.get("uri"),
                        "gemini_mime_type": uploaded.get("mime_type"),
                    },
                )
            except Exception as exc:
                logger.warning("File upload failed for %s: %s", filename, exc)
                await log_error(
                    "chat",
                    "chat_agent.upload_file_to_gemini",
                    f"Gemini file upload failed: {filename}",
                    user_id=stream_user_id,
                    guest_session_id=guest_session_id,
                    endpoint="/chat/message",
                    exc=exc,
                    http_status=502,
                )
                yield chat_agent._sse_error(
                    "File upload failed. Please try again or use a different file."
                )
                return

            file_type = _gemini_type_from_mime(uploaded["mime_type"])
            file_parts.append({
                "type": file_type,
                "uri": uploaded["uri"],
                "mime_type": uploaded["mime_type"],
            })

        input_parts: list = []
        if message:
            input_parts.append({"type": "text", "text": message})
        input_parts.extend(file_parts)

        async for chunk in chat_agent.stream_chat(
            input_parts=input_parts,
            previous_interaction_id=effective_prev or None,
            user_id=stream_user_id,
            guest_session_id=guest_session_id,
        ):
            yield chunk

    return StreamingResponse(
        _stream_with_session_save(
            stream_user_id,
            _produce_message_stream(),
            guest_session_id=guest_session_id,
            guest_session_row=guest_session_row,
            db=db if guest_session_row else None,
            user_text=memory_user or None,
            increment_guest_on_done=guest_session_row is not None,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ── POST /chat/tool_result ────────────────────────────────────────────────────

@router.post("/tool_result")
async def chat_tool_result(
    request: Request,
    body: ToolResultRequest,
    current_user: User | None = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Submit tool execution results and receive the model's follow-up response.

    Call this immediately after the frontend has executed all tool calls
    received from a previous /chat/message or /chat/tool_result stream.

    Request body (JSON):
    {
      "previous_interaction_id": "v1_...",
      "results": [
        {
          "call_id": "fc_...",
          "name":    "AISearch",
          "result":  [...],     // any JSON value
          "error":   false
        }
      ]
    }

    Returns the same SSE stream format as /chat/message.
    The response may contain additional tool calls for multi-step operations.
    """
    if not body.results:
        raise HTTPException(status_code=422, detail="results must not be empty.")

    guest_session_id: str | None = None
    if current_user is None:
        guest_session_id = require_guest_session_id(request)
        # Match /chat/message: fingerprint remapping may change the canonical UUID.
        guest_identity = build_guest_request_identity(
            request.headers,
            peer_host=request.client.host if request.client else None,
            raw_fingerprint=request.headers.get("x-guest-fingerprint"),
        )
        guest_row = await get_guest_session_row(
            db,
            guest_session_id,
            guest_identity.ip_hash,
            fingerprint_hash=guest_identity.fingerprint_hash,
            composite_hash=guest_identity.composite_hash,
            subnet_key=guest_identity.subnet_key,
            ja4_fingerprint=guest_identity.ja4_fingerprint,
            asn=guest_identity.asn,
        )
        guest_session_id = guest_row.session_id

    stream_user_id = current_user.id if current_user else None
    if current_user is not None:
        chain_prev = await chat_session.get_interaction_id(current_user.id) or body.previous_interaction_id
    else:
        chain_prev = (
            await chat_session.get_guest_interaction_id(guest_session_id)
            or body.previous_interaction_id
        )

    chat_agent._dbg(
        "[CHAT ROUTER ▶ TOOL RESULT]",
        {
            "user_id": stream_user_id,
            "guest_session_id": guest_session_id,
            "previous_interaction_id": body.previous_interaction_id,
            "results": [
                {
                    "name": item.name,
                    "call_id": item.call_id,
                    "error": item.error,
                    "result_chars": len(json.dumps(item.result, ensure_ascii=False)),
                    "result_preview": json.dumps(item.result, ensure_ascii=False)[:1000],
                }
                for item in body.results
            ],
        },
    )

    # Convert to Gemini function_result input parts
    input_parts: list[dict] = []
    for item in body.results:
        result_payload: object
        if item.error:
            result_payload = [{"type": "text", "text": f"Error: {json.dumps(item.result)}"}]
        else:
            compacted = _compact_tool_result(item.name, item.result)
            result_payload = [{"type": "text", "text": json.dumps(compacted, ensure_ascii=False)}]

        input_parts.append({
            "type":    "function_result",
            "name":    item.name,
            "call_id": item.call_id,
            "result":  result_payload,
        })

    return StreamingResponse(
        _stream_with_session_save(
            stream_user_id,
            chat_agent.stream_chat(
                input_parts=input_parts,
                previous_interaction_id=chain_prev,
                user_id=stream_user_id,
                guest_session_id=guest_session_id,
            ),
            guest_session_id=guest_session_id,
            user_text=", ".join(f"[Tool: {item.name}]" for item in body.results),
            record_transcript=False,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


def _compact_tool_result(tool_name: str, result: object) -> object:
    """Reduce very large frontend tool payloads before they enter model context."""
    if tool_name == "AISearch":
        return _compact_ai_search_result(result)

    if tool_name == "KeywordSearch":
        return _compact_keyword_search_result(result)

    if tool_name in ("open_ai_generations", "show_image_generations"):
        return _compact_generations_list(result)

    if tool_name == "generate_multi_image":
        return _compact_generate_multi_image_result(result)

    raw = json.dumps(result, ensure_ascii=False)
    _mtc = _max_tool_result_chars()
    if len(raw) <= _mtc:
        return result
    return {
        "_truncated": True,
        "_original_chars": len(raw),
        "summary": raw[:_mtc],
    }


def _compact_generate_multi_image_result(result: object) -> object:
    """Keep generation ids/refs for the model; drop long signed URLs."""
    if not isinstance(result, dict):
        return result
    gid = result.get("id") or result.get("generation_id")
    compact: dict = {
        "_type": "GenerateMultiImageResult",
        "_compacted_for_model": True,
        "id": gid,
        "generation_id": gid,
        "generation_name": result.get("generation_name"),
        "generation_ref": f"gen:{gid}" if gid else result.get("generation_ref"),
        "model_used": result.get("model_used"),
        "images_used": result.get("images_used"),
        "approved": result.get("approved"),
        "input_quality": result.get("input_quality"),
        "generation_warnings": result.get("generation_warnings"),
    }
    # Preserve a short URL hint only if already short (avoid megabyte data-URIs).
    for key in ("url", "signed_url"):
        val = result.get(key)
        if isinstance(val, str) and val.startswith("http") and len(val) <= 512:
            compact[key] = val
    return compact


def _compact_generations_list(result: object) -> object:
    """Compact generation lists so the model gets ids/refs without megabytes of signed URLs."""
    if isinstance(result, dict) and isinstance(result.get("generations"), list):
        items = result["generations"]
        note = result.get("note")
    elif isinstance(result, list):
        items = result
        note = None
    else:
        items = []
        note = None
    compact: list[dict] = []
    for item in items[:12]:
        if not isinstance(item, dict):
            continue
        gid = item.get("id")
        compact.append({
            "id": gid,
            "generation_name": item.get("generation_name"),
            "generation_ref": f"gen:{gid}" if gid else None,
            "generated_at": item.get("generated_at"),
            "name": item.get("name") or item.get("generation_name"),
        })
    out: dict = {
        "_type": "GenerationsList",
        "_compacted_for_model": True,
        "_original_count": len(items),
        "items": compact,
        "instruction": (
            "Call get_image(generation_ref) or get_image(id) to open a specific generation. "
            "Use generation_ref (e.g. gen:339) to avoid catalog id collisions."
        ),
    }
    if note:
        out["note"] = note
    return out


def _compact_ai_search_result(result: object) -> object:
    items = result if isinstance(result, list) else []
    _mai = _max_aisearch_items()
    compact_items: list[dict] = []
    for item in items[:_mai]:
        if not isinstance(item, dict):
            continue
        compact_items.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "filename": item.get("filename"),
            "class": item.get("class"),
            "theme": item.get("theme"),
            "project": item.get("project"),
            "level": item.get("level"),
            "tags": (item.get("tags") or [])[:20] if isinstance(item.get("tags"), list) else item.get("tags"),
            "url": item.get("url") or item.get("link"),
            "_type": item.get("_type") or "catalog_image",
            "description": _truncate_text(item.get("description"), 700),
        })
    return {
        "_type": "AISearchResult",
        "_compacted_for_model": True,
        "_original_count": len(items),
        "items": compact_items,
        "instruction": (
            "Results are already shown to the user in a labeled chat slider. "
            "If you also ran KeywordSearch for products, both sliders are visible — "
            "do NOT repeat searches. Do NOT call get_image unless the user asks."
        ),
    }


def _compact_keyword_search_result(result: object) -> object:
    """Compact keyword search JSON from chat executor."""
    parsed = result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            parsed = result

    if isinstance(parsed, dict) and "designs" in parsed:
        items = parsed.get("designs") or []
        return {
            "_type": "KeywordSearchResult",
            "_compacted_for_model": True,
            "found": parsed.get("found", len(items)),
            "mode": parsed.get("mode", "keyword"),
            "items": items[:_max_aisearch_items()],
            "instruction": (
                "Catalog filter results are shown in a separate labeled slider. "
                "If AISearch was also called, both result sets are already visible."
            ),
        }

    items = parsed if isinstance(parsed, list) else []
    compact_items: list[dict] = []
    for item in items[:_max_aisearch_items()]:
        if not isinstance(item, dict):
            continue
        compact_items.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "filename": item.get("filename"),
            "class": item.get("class"),
            "level": item.get("level"),
            "_type": item.get("_type") or "catalog_image",
        })
    return {
        "_type": "KeywordSearchResult",
        "_compacted_for_model": True,
        "_original_count": len(items),
        "items": compact_items,
        "instruction": (
            "Catalog filter results are shown in a separate labeled slider. "
            "If AISearch was also called, both result sets are already visible."
        ),
    }


def _truncate_text(value: object, max_chars: int) -> object:
    if not isinstance(value, str):
        return value
    return value if len(value) <= max_chars else value[:max_chars] + "..."


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_mime(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith((".heic", ".heif")):
        return "image/heic"
    if lower.endswith(".pdf"):
        return "application/pdf"
    if lower.endswith(".mp4"):
        return "video/mp4"
    if lower.endswith(".mp3"):
        return "audio/mpeg"
    if lower.endswith(".wav"):
        return "audio/wav"
    return "application/octet-stream"


# Anthropic accepts only these image media types inline.
_CLAUDE_IMAGE_MIMES = ("image/jpeg", "image/png", "image/gif", "image/webp")


def _normalize_image_for_claude(data: bytes, mime: str) -> tuple[str, bytes]:
    """
    Return (media_type, bytes) suitable for Anthropic. JPEG/PNG/GIF/WEBP pass
    through untouched; anything else (HEIC/HEIF/TIFF/…) is decoded with Pillow
    (HEIF opener is registered in main.py) and re-encoded to JPEG.
    """
    if mime in _CLAUDE_IMAGE_MIMES:
        return mime, data
    try:
        from PIL import Image as _PILImage
        import io as _io
        img = _PILImage.open(_io.BytesIO(data))
        buf = _io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=92)
        return "image/jpeg", buf.getvalue()
    except Exception as exc:
        logger.warning("Could not normalize %s for Claude (%s); sending as-is.", mime, exc)
        return mime, data


def _gemini_type_from_mime(mime: str) -> str:
    """Map a MIME type to the Gemini Interactions API content type string."""
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("video/"):
        return "video"
    if mime.startswith("audio/"):
        return "audio"
    if mime == "application/pdf":
        return "document"
    return "document"


# ── POST /chat/voice_context ──────────────────────────────────────────────────

class VoiceContextRequest(BaseModel):
    """
    Inject a completed voice-agent turn into the shared Gemini Interactions
    history so that subsequent text-chat turns are aware of it.

    Send this after each voice turn that included meaningful content (user spoke
    + model responded, and/or a tool like start_designer_job completed).

    The backend makes ONE real Gemini Interactions call (user=voice_summary,
    model=acknowledge) and returns the new interaction_id.  The frontend should
    store this as its next previous_interaction_id so the next text-chat turn
    chains from here.
    """
    user_text:               str = ""
    assistant_text:          str = ""
    tool_name:               Optional[str]   = None
    tool_summary:            Optional[str]   = None   # compact JSON-friendly text
    previous_interaction_id: Optional[str]   = None


@router.post("/voice_context")
async def inject_voice_context(
    body: VoiceContextRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Inject a completed voice turn into the Gemini Interactions history chain.
    Returns { "interaction_id": "v1_..." } — store and pass as
    previous_interaction_id on the next /chat/message call.
    """
    # Store voice turns in the shared Interactions history.
    # For normal dialogue, pass the user's exact words as the user turn (best recall).
    if body.tool_name and body.tool_summary:
        parts: list[str] = []
        if body.user_text:
            parts.append(f"User: {body.user_text}")
        if body.assistant_text:
            parts.append(f"Assistant (voice): {body.assistant_text}")
        parts.append(f"Tool executed: {body.tool_name}")
        if body.tool_name == "start_designer_job":
            parts.append("Designer job result — store ALL of the following for follow-up questions:")
            parts.append(body.tool_summary)
            parts.append(
                "IMPORTANT: When the user asks what Ducon images, references, or catalog "
                "images were used in this design, answer ONLY from the reference list above. "
                "Do not call AISearch or search the catalog again for that question. "
                "Use get_image(catalog_id) or get_image(generation_ref) to open specific images."
            )
        else:
            parts.append(f"Result summary: {body.tool_summary}")
        parts.append(
            "This exchange happened in the voice session. Remember it for follow-up text chat. "
            "Acknowledge in one short sentence (e.g. 'Noted.')."
        )
        input_parts = [{"type": "text", "text": "\n".join(parts)}]
    elif body.user_text:
        memory_text = body.user_text
        if body.assistant_text:
            memory_text += (
                "\n\n[Voice assistant reply for context: "
                f"{body.assistant_text}]"
            )
        input_parts = [{"type": "text", "text": memory_text}]
    elif body.assistant_text:
        input_parts = [{
            "type": "text",
            "text": (
                f"[Voice assistant said: {body.assistant_text}]\n"
                "Remember this from the voice session for follow-up text chat. "
                "Acknowledge in one short sentence."
            ),
        }]
    else:
        return {"interaction_id": await chat_session.get_interaction_id(current_user.id)}

    session_prev = await chat_session.get_interaction_id(current_user.id)
    chain_prev = session_prev or body.previous_interaction_id

    # Normal voice turns are already stored by GeminiLiveSession._commit_turn.
    # Only async designer-job memory injects need an extra transcript row here.
    record_transcript = body.tool_name == "start_designer_job"
    memory_user = None
    if record_transcript:
        memory_user = (input_parts[0].get("text") if input_parts else None)

    interaction_id: Optional[str] = None
    async for raw in _stream_with_session_save(
        current_user.id,
        chat_agent.stream_chat(
            input_parts=input_parts,
            previous_interaction_id=chain_prev,
            allow_tools=False,
            user_id=current_user.id,
        ),
        user_text=memory_user,
        record_transcript=record_transcript,
    ):
        # The SSE stream contains JSON events — we only care about `done`
        if not raw.startswith("data:"):
            continue
        payload_str = raw[len("data:"):].strip()
        try:
            payload = json.loads(payload_str)
            if payload.get("type") == "done":
                interaction_id = payload.get("interaction_id")
                break
        except (json.JSONDecodeError, AttributeError):
            continue

    return {"interaction_id": interaction_id}


# ── POST /chat/browse_context ─────────────────────────────────────────────────

class BrowseContextRequest(BaseModel):
    """
    Inject UI-driven catalog browse steps (explore products, category picks, etc.)
    into the shared Interactions history without sending image data.
    """
    actions: list[dict]
    previous_interaction_id: Optional[str] = None


def _format_product_ref(idx: int, item: dict) -> str:
    cid = item.get("catalog_id")
    name = item.get("name") or f"catalog {cid}"
    desc = item.get("description") or ""
    filename = item.get("filename")
    parts = [f"     {idx}. catalog_id={cid} name=\"{name}\""]
    if desc:
        parts.append(f"desc=\"{desc}\"")
    if filename:
        parts.append(f"filename=\"{filename}\"")
    return " ".join(parts)


def _format_browse_memory(actions: list[dict]) -> str:
    lines = [
        "[Chat UI browse session — the user explored catalog options in the chat UI "
        "(not via agent tools). No images are included.",
        "Use catalog_id with get_image(catalog_id) to open a catalog image, or "
        "generation_ref with get_image(generation_ref) for AI generations.",
        "Sequence:",
    ]
    step = 1
    for action in actions:
        kind = action.get("type") or ""
        if kind == "explore_products":
            lines.append(f"{step}. User chose to explore Ducon products.")
            user_msg = action.get("user_message")
            if user_msg:
                lines.append(f"   User prompt shown: \"{user_msg}\"")
            cats = action.get("categories") or []
            if cats:
                lines.append(f"   Product categories shown: {', '.join(cats)}")
            step += 1
        elif kind == "category_selected":
            cat = action.get("category") or "unknown"
            lines.append(f"{step}. User selected product category \"{cat}\".")
            products = action.get("products") or []
            if products:
                lines.append(f"   Products shown ({len(products)}):")
                for i, p in enumerate(products, 1):
                    lines.append(_format_product_ref(i, p))
            step += 1
        elif kind == "product_viewed":
            cat = action.get("category")
            prefix = f"{step}. User opened product"
            if cat:
                prefix += f" from \"{cat}\""
            lines.append(f"{prefix} for preview:")
            lines.append(f"   {_format_product_ref(1, action).lstrip()}")
            step += 1
        elif kind == "catalog_attached":
            lines.append(f"{step}. User attached a catalog reference to chat:")
            lines.append(f"   {_format_product_ref(1, action).lstrip()}")
            step += 1
        elif kind == "generation_attached":
            gen_ref = action.get("generation_ref") or f"gen:{action.get('generation_id')}"
            name = action.get("name") or f"generation {action.get('generation_id')}"
            lines.append(
                f"{step}. User attached AI generation \"{name}\" "
                f"(generation_ref=\"{gen_ref}\", id={action.get('generation_id')})."
            )
            step += 1
        else:
            summary = action.get("summary") or json.dumps(action, ensure_ascii=False)
            lines.append(f"{step}. {summary}")
            step += 1

    lines.append(
        "Remember this browse context for follow-up questions. "
        "Acknowledge in one short sentence (e.g. 'Noted your product browsing.')."
    )
    return "\n".join(lines)


@router.post("/browse_context")
async def inject_browse_context(
    body: BrowseContextRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Persist UI browse steps into Gemini Interactions history (text-only refs).
    Returns { "interaction_id": "v1_..." } for the next /chat/message call.
    """
    if not body.actions:
        return {"interaction_id": await chat_session.get_interaction_id(current_user.id)}

    memory_text = _format_browse_memory(body.actions)
    input_parts = [{"type": "text", "text": memory_text}]

    session_prev = await chat_session.get_interaction_id(current_user.id)
    chain_prev = session_prev or body.previous_interaction_id

    interaction_id: Optional[str] = None
    async for raw in _stream_with_session_save(
        current_user.id,
        chat_agent.stream_chat(
            input_parts=input_parts,
            previous_interaction_id=chain_prev,
            allow_tools=False,
            user_id=current_user.id,
        ),
        user_text=(input_parts[0].get("text") if input_parts else None),
    ):
        if not raw.startswith("data:"):
            continue
        payload_str = raw[len("data:"):].strip()
        try:
            payload = json.loads(payload_str)
            if payload.get("type") == "done":
                interaction_id = payload.get("interaction_id")
                break
        except (json.JSONDecodeError, AttributeError):
            continue

    return {"interaction_id": interaction_id}


# ── POST /chat/studio_context ────────────────────────────────────────────────

class StudioContextRequest(BaseModel):
    """
    Inject Studio AI wizard context (selections, prompt, generation ref) into
    the shared Interactions history. Images are attached separately via /chat/message.
    """
    context: dict
    previous_interaction_id: Optional[str] = None


def _format_studio_memory(ctx: dict) -> str:
    gen_id = ctx.get("generation_id")
    gen_ref = ctx.get("generation_ref") or (f"gen:{gen_id}" if gen_id else None)
    gen_name = ctx.get("generation_name") or f"generation {gen_id}"
    prompt = (ctx.get("prompt") or "").strip()

    lines = [
        "[Studio AI visualization — the user completed the Studio wizard and generated a design.",
        "The next user message includes attached images: studio result, original space photo, "
        "and catalog reference images where available.",
        "Use generation_ref with get_image(generation_ref) to reopen the AI result; "
        "catalog_id with get_image(catalog_id) for catalog references.",
        "",
        "Generation:",
        f"  generation_id={gen_id} generation_ref=\"{gen_ref}\" name=\"{gen_name}\"",
    ]

    if prompt:
        lines.append("")
        lines.append("Prompt used for generation:")
        lines.append(prompt)

    space_summary = ctx.get("space_summary")
    style_summary = ctx.get("style_summary")
    direction = ctx.get("direction_title")
    if space_summary:
        lines.append(f"Space type(s): {space_summary}")
    if style_summary:
        lines.append(f"Style direction(s): {style_summary}")
    if direction:
        lines.append(f"Design direction: {direction}")

    ref = ctx.get("reference_design")
    if ref and ref.get("catalog_id"):
        lines.append(
            f"Reference design: catalog_id={ref['catalog_id']} "
            f"name=\"{ref.get('name') or 'reference'}\""
        )

    products = ctx.get("products") or []
    if products:
        lines.append(f"Products included ({len(products)}):")
        for i, p in enumerate(products, 1):
            cid = p.get("catalog_id")
            label = p.get("label") or p.get("name") or f"product {i}"
            if cid:
                lines.append(f"  {i}. catalog_id={cid} label=\"{label}\"")
            else:
                lines.append(f"  {i}. label=\"{label}\"")

    lines.append("")
    lines.append(
        "Remember this Studio context for follow-up edits. "
        "Acknowledge briefly and offer concrete refinement options "
        "(materials, layout, products, lighting, etc.)."
    )
    return "\n".join(lines)


@router.post("/studio_context")
async def inject_studio_context(
    body: StudioContextRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Persist Studio wizard generation context into Gemini Interactions history.
    Returns { "interaction_id": "v1_..." } for the next /chat/message call.
    """
    if not body.context:
        return {"interaction_id": await chat_session.get_interaction_id(current_user.id)}

    memory_text = _format_studio_memory(body.context)
    input_parts = [{"type": "text", "text": memory_text}]

    session_prev = await chat_session.get_interaction_id(current_user.id)
    chain_prev = session_prev or body.previous_interaction_id

    interaction_id: Optional[str] = None
    async for raw in _stream_with_session_save(
        current_user.id,
        chat_agent.stream_chat(
            input_parts=input_parts,
            previous_interaction_id=chain_prev,
            allow_tools=False,
            user_id=current_user.id,
        ),
        user_text=(input_parts[0].get("text") if input_parts else None),
    ):
        if not raw.startswith("data:"):
            continue
        payload_str = raw[len("data:"):].strip()
        try:
            payload = json.loads(payload_str)
            if payload.get("type") == "done":
                interaction_id = payload.get("interaction_id")
                break
        except (json.JSONDecodeError, AttributeError):
            continue

    return {"interaction_id": interaction_id}
