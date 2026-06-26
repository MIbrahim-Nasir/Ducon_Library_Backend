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

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.auth import get_current_user
from app.db.models import User
from app import chat_agent
from app import chat_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])
_MAX_TOOL_RESULT_CHARS = int(os.getenv("CHAT_TOOL_RESULT_MAX_CHARS", "12000"))
_MAX_AISEARCH_ITEMS = int(os.getenv("CHAT_AISEARCH_MAX_ITEMS", "8"))
_MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50")) * 1024 * 1024

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
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {_MAX_UPLOAD_BYTES // (1024*1024)} MB.",
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

_SSE_HEADERS = {
    "Cache-Control":     "no-cache",
    "Connection":        "keep-alive",
    "Content-Encoding":  "identity",   # prevent any intermediate compression
    "X-Accel-Buffering": "no",         # disable nginx / Cloudflare buffering
}


async def _stream_with_session_save(
    user_id: int,
    generator,
):
    """Persist interaction_id and emit keepalive pings to prevent proxy timeouts."""
    KEEPALIVE_INTERVAL = 15.0  # seconds — resets Cloudflare's 100 s idle timeout

    async def _next_or_timeout(aiter, timeout):
        """Return (chunk, False) or (None, True) on timeout."""
        try:
            return await asyncio.wait_for(aiter.__anext__(), timeout=timeout), False
        except StopAsyncIteration:
            return None, False
        except asyncio.TimeoutError:
            return None, True

    aiter = generator.__aiter__()
    while True:
        chunk, timed_out = await _next_or_timeout(aiter, KEEPALIVE_INTERVAL)
        if timed_out:
            yield ": keepalive\n\n"
            continue
        if chunk is None:
            break
        if chunk.startswith("data:"):
            try:
                payload = json.loads(chunk[5:].strip())
                if payload.get("type") == "done" and payload.get("interaction_id"):
                    chat_session.set_interaction_id(user_id, payload["interaction_id"])
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        yield chunk


# ── GET /chat/session ─────────────────────────────────────────────────────────

@router.get("/session")
async def get_chat_session(current_user: User = Depends(get_current_user)):
    """Return the persisted Interactions API chain id for this user (voice + text share it)."""
    return {"interaction_id": chat_session.get_interaction_id(current_user.id)}


@router.delete("/session", status_code=204)
async def clear_chat_session(current_user: User = Depends(get_current_user)):
    """Clear the stored chat session (e.g. when user clears the conversation)."""
    chat_session.clear_session(current_user.id)


# ── POST /chat/message ────────────────────────────────────────────────────────

@router.post("/message")
async def chat_message(
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
    files: List[UploadFile] = File(
        default=[],
        description="Images or documents to attach to this message.",
    ),
    current_user: User = Depends(get_current_user),
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
    chat_agent._dbg(
        "[CHAT ROUTER ▶ MESSAGE]",
        {
            "user_id": current_user.id,
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

    # ── Upload attached files to Gemini Files API ─────────────────────────────
    file_parts: list[dict] = []
    for upload in files:
        if not upload.filename:
            continue
        file_bytes = await upload.read()
        if not file_bytes:
            continue
        _validate_upload(upload.filename, file_bytes)
        mime = upload.content_type or _infer_mime(upload.filename)
        try:
            uploaded = await chat_agent.upload_file_to_gemini(
                file_bytes=file_bytes,
                filename=upload.filename,
                mime_type=mime,
            )
            chat_agent._dbg(
                "[CHAT ROUTER ▶ FILE UPLOADED]",
                {
                    "filename": upload.filename,
                    "bytes": len(file_bytes),
                    "mime_type": mime,
                    "gemini_uri": uploaded.get("uri"),
                    "gemini_mime_type": uploaded.get("mime_type"),
                },
            )
        except Exception as exc:
            logger.warning("File upload failed for %s: %s", upload.filename, exc)
            raise HTTPException(
                status_code=502,
                detail="File upload failed. Please try again or use a different file.",
            ) from exc

        file_type = _gemini_type_from_mime(uploaded["mime_type"])
        file_parts.append({
            "type":      file_type,
            "uri":       uploaded["uri"],
            "mime_type": uploaded["mime_type"],
        })

    # ── Build input parts ─────────────────────────────────────────────────────
    input_parts: list = []
    if message:
        input_parts.append({"type": "text", "text": message})
    input_parts.extend(file_parts)

    # Prefer server-persisted session (updated by voice_context injects) over a
    # possibly stale client previous_interaction_id.
    session_prev = chat_session.get_interaction_id(current_user.id)
    effective_prev = session_prev or previous_interaction_id
    return StreamingResponse(
        _stream_with_session_save(
            current_user.id,
            chat_agent.stream_chat(
                input_parts=input_parts,
                previous_interaction_id=effective_prev or None,
            ),
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ── POST /chat/tool_result ────────────────────────────────────────────────────

@router.post("/tool_result")
async def chat_tool_result(
    body: ToolResultRequest,
    current_user: User = Depends(get_current_user),
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
    chat_agent._dbg(
        "[CHAT ROUTER ▶ TOOL RESULT]",
        {
            "user_id": current_user.id,
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
            current_user.id,
            chat_agent.stream_chat(
                input_parts=input_parts,
                previous_interaction_id=(
                    chat_session.get_interaction_id(current_user.id)
                    or body.previous_interaction_id
                ),
            ),
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


def _compact_tool_result(tool_name: str, result: object) -> object:
    """Reduce very large frontend tool payloads before they enter model context."""
    if tool_name == "AISearch":
        return _compact_ai_search_result(result)

    if tool_name in ("open_ai_generations", "show_image_generations"):
        return _compact_generations_list(result)

    raw = json.dumps(result, ensure_ascii=False)
    if len(raw) <= _MAX_TOOL_RESULT_CHARS:
        return result
    return {
        "_truncated": True,
        "_original_chars": len(raw),
        "summary": raw[:_MAX_TOOL_RESULT_CHARS],
    }


def _compact_generations_list(result: object) -> object:
    """Compact generation lists so the model gets ids/refs without megabytes of signed URLs."""
    items = result if isinstance(result, list) else []
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
    return {
        "_type": "GenerationsList",
        "_compacted_for_model": True,
        "_original_count": len(items),
        "items": compact,
        "instruction": (
            "Call get_image(generation_ref) or get_image(id) to open a specific generation. "
            "Use generation_ref (e.g. gen:339) to avoid catalog id collisions."
        ),
    }


def _compact_ai_search_result(result: object) -> object:
    items = result if isinstance(result, list) else []
    compact_items: list[dict] = []
    for item in items[:_MAX_AISEARCH_ITEMS]:
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
            "Results are already shown to the user in the chat image slider. "
            "Do NOT call get_image or run additional AISearch/KeywordSearch unless "
            "the user asks for more. Catalog ids require plain id; AI generations use gen:ID."
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
        return {"interaction_id": chat_session.get_interaction_id(current_user.id)}

    session_prev = chat_session.get_interaction_id(current_user.id)
    chain_prev = session_prev or body.previous_interaction_id

    interaction_id: Optional[str] = None
    async for raw in _stream_with_session_save(
        current_user.id,
        chat_agent.stream_chat(
            input_parts=input_parts,
            previous_interaction_id=chain_prev,
        ),
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
        return {"interaction_id": chat_session.get_interaction_id(current_user.id)}

    memory_text = _format_browse_memory(body.actions)
    input_parts = [{"type": "text", "text": memory_text}]

    session_prev = chat_session.get_interaction_id(current_user.id)
    chain_prev = session_prev or body.previous_interaction_id

    interaction_id: Optional[str] = None
    async for raw in _stream_with_session_save(
        current_user.id,
        chat_agent.stream_chat(
            input_parts=input_parts,
            previous_interaction_id=chain_prev,
        ),
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
        return {"interaction_id": chat_session.get_interaction_id(current_user.id)}

    memory_text = _format_studio_memory(body.context)
    input_parts = [{"type": "text", "text": memory_text}]

    session_prev = chat_session.get_interaction_id(current_user.id)
    chain_prev = session_prev or body.previous_interaction_id

    interaction_id: Optional[str] = None
    async for raw in _stream_with_session_save(
        current_user.id,
        chat_agent.stream_chat(
            input_parts=input_parts,
            previous_interaction_id=chain_prev,
        ),
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
