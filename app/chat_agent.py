"""
Ducon Designer Chat Agent
=========================
Stateful multimodal chat agent using the Gemini Interactions API.

Features
--------
- Multi-turn conversation with server-managed history (previous_interaction_id)
- Streaming text/thinking via SSE
- Multimodal input (text + images via Gemini Files API)
- Frontend tool calling (same tools as the voice agent)
- Long-running tasks (AI generation, quotation)

Architecture
------------
Each user turn is one HTTP request → SSE stream. When the model requests a
tool call, the stream ends with a `tool_call` event and the current
`interaction_id`.  The frontend executes the tool and submits the result to
POST /chat/tool_result, which opens a new SSE stream for the model's response.
Multi-turn tool chains (sequential calls) work naturally through this mechanism.

Streaming SSE event types (emitted to frontend)
-----------------------------------------------
  {"type": "text_delta",    "text": "..."}         → incremental response text
  {"type": "thinking_delta","text": "..."}         → model reasoning (if enabled)
  {"type": "tool_call",     "id":"...","name":"...","args":{...}}  → execute this
  {"type": "done",          "interaction_id": "..."} → turn complete; save this ID
  {"type": "error",         "message": "..."}       → something went wrong
"""
from __future__ import annotations

import io
import json
import logging
import os
import asyncio
import tempfile
import uuid
from typing import AsyncGenerator, Optional

from google import genai

from app import llm_provider
from app.search_tools import ai_search_interactions_tool, keyword_search_interactions_tool

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
from app.admin.settings_store import cfg, cfg_str
from app.error_logger import log_error

CHAT_MODEL = "gemini-3.5-flash"                      # default; live value via cfg("CHAT_MODEL", CHAT_MODEL)
CHAT_THINKING_LEVEL = "low"
# Set CHAT_STREAM to "false" to disable streaming (useful for debugging).
CHAT_STREAM: bool = True
_LIVE_DEBUG: bool = False


def _dbg(*args) -> None:
    if cfg("LIVE_DEBUG", _LIVE_DEBUG):
        print(*args)


def _summarize_input_parts(parts: list) -> list[dict]:
    summary: list[dict] = []
    for part in parts:
        if isinstance(part, str):
            summary.append({"type": "text", "chars": len(part), "preview": part[:240]})
        elif isinstance(part, dict):
            ptype = part.get("type")
            if ptype == "text":
                text = part.get("text", "")
                summary.append({"type": "text", "chars": len(text), "preview": text[:240]})
            elif ptype in ("image", "audio", "video", "document"):
                summary.append({
                    "type": ptype,
                    "uri": part.get("uri"),
                    "mime_type": part.get("mime_type"),
                })
            elif ptype == "function_result":
                result = part.get("result")
                summary.append({
                    "type": "function_result",
                    "name": part.get("name"),
                    "call_id": part.get("call_id"),
                    "result_chars": len(json.dumps(result, ensure_ascii=False)) if result is not None else 0,
                })
            else:
                summary.append({"type": ptype or "dict", "keys": list(part.keys())})
        else:
            summary.append({"type": type(part).__name__})
    return summary


def _event_usage(event) -> dict | None:
    metadata = getattr(event, "metadata", None)
    if not metadata:
        return None
    if hasattr(metadata, "model_dump"):
        data = metadata.model_dump(exclude_none=True)
    elif isinstance(metadata, dict):
        data = metadata
    else:
        return None
    return data.get("total_usage") or data.get("usage")


def _interaction_usage(interaction) -> dict | None:
    usage = getattr(interaction, "usage", None)
    if usage is None:
        return None
    return usage.model_dump(exclude_none=True) if hasattr(usage, "model_dump") else usage


def _record_usage(usage: dict | None, *, model: str, provider: str = "gemini",
                  user_id: Optional[int] = None, guest_session_id: Optional[str] = None,
                  status: str = "success", error_message: Optional[str] = None) -> None:
    """Non-blocking usage/cost recording. Best-effort; never raises."""
    if usage is None:
        # Still record a zero-token event so call volume is tracked.
        try:
            from app.admin.usage_recorder import record
            record(agent="chat", model=model, provider=provider,
                   user_id=user_id, guest_session_id=guest_session_id,
                   status=status, error_message=error_message)
        except Exception:
            pass
        return
    try:
        from app.admin.usage_recorder import record
        record(
            agent="chat", model=model, provider=provider,
            user_id=user_id, guest_session_id=guest_session_id,
            input_tokens=int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or usage.get("candidates_tokens") or 0),
            status=status, error_message=error_message,
        )
    except Exception:
        pass

# ── Gemini client (shared singleton) ─────────────────────────────────────────

_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


from app import prompt_loader


def get_chat_system_instruction() -> str:
    override = os.getenv("CHAT_SYSTEM_INSTRUCTION")
    if override:
        return override.strip()
    prompt_loader.ensure_prompts_loaded()
    return prompt_loader.CHAT_AGENT_SYSTEM


# ── Tool declarations (Interactions API format) ───────────────────────────────
CHAT_TOOLS: list[dict] = [
    ai_search_interactions_tool(),
    keyword_search_interactions_tool(require_query=False, for_chat=True),
    {
        "type": "function",
        "name": "get_selected_image",
        "description": (
            "Returns the currently open/selected image, or null. Usually returns CatalogImage. "
            "Call this proactively when the user refers to 'this image', 'this design', "
            "'something like this', or implies they are looking at something. "
            "Do not ask the user which image they mean until you have tried this first."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "get_image",
        "description": (
            "Resolve a catalog image, AI generation, or user upload by reference. "
            "In chat, results appear inline — do NOT use after AISearch browse flows. "
            "Use only when generation/quotation needs a concrete ref. "
            "Catalog: numeric id or name. AI generation: always gen:ID (ids collide with catalog)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "Image reference: numeric ID, name/filename, generation ref, upload ref, or JSON object with id.",
                },
                "show_user": {
                    "type": "boolean",
                    "description": "True to open the image viewer for the user. False when fetching for internal analysis only.",
                },
            },
            "required": ["ref"],
        },
    },
    {
        "type": "function",
        "name": "show_user_uploaded_images",
        "description": "Open the Uploads panel in the sidebar showing the user's previously uploaded photos.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "show_image_generations",
        "description": "Open the AI Generations panel in the sidebar showing completed design previews.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "show_bookmarks",
        "description": "Open the Bookmarks panel in the sidebar showing items the user has saved.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "open_sidebar",
        "description": (
            "Open the sidebar workspace panel. Returns {status:'open'}. "
            "Use when the user asks to open the sidebar or wants to access their workspace."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "open_ai_generations",
        "description": (
            "Open the AI Generations panel in the sidebar and return the list of the user's "
            "AI-generated designs. Returns Array<{id, generation_name, url, ducon_image_id}>. "
            "Use when the user asks to see, browse, or access their generated images. "
            "After generate_multi_image you can call this to show the user their result "
            "in context with past generations."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "start_designer_job",
        "description": (
            "Start a long-running autonomous Ducon design job. Use when the user provides a "
            "client space image and wants the designer to explore ideas, search multiple Ducon "
            "references, generate one or more candidates, visually evaluate them, and return the "
            "best result. This is preferred for requests like 'design this', 'make my terrace look "
            "good', 'try a few ideas', or any task that may need multiple searches/generations. "
            "The frontend must resolve user_upload_image to an actual File and call POST "
            "/designer/jobs. Returns {job_id, status, events_url}; progress arrives from the job "
            "events stream."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "User's design goal or suggestions. If the user gave no instructions, "
                        "summarize that the designer should choose an appropriate Ducon concept."
                    ),
                },
                "user_upload_image": {
                    "type": "string",
                    "description": (
                        "Upload id/source for the client's space image. Use an upload id from a "
                        "chat attachment message. If missing, ask the user to attach their space "
                        "photo with the paperclip button in the chat composer."
                    ),
                },
                "model": {
                    "type": "string",
                    "enum": ["flash", "pro"],
                    "description": "Image generation model for attempts. Default: flash.",
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": ["1:1", "4:3", "3:4", "16:9", "9:16"],
                    "description": "Output aspect ratio. Default: 16:9.",
                },
                "show_user": {
                    "type": "boolean",
                    "description": "Usually false — autonomous design runs in the chat timeline, not separate UI.",
                },
            },
            "required": ["prompt", "user_upload_image"],
        },
    },
    {
        "type": "function",
        "name": "generate_multi_image",
        "description": (
            "Generate a new design image by compositing multiple source images with a custom prompt. "
            "Use this for both simple single-reference generations and advanced multi-image tasks. "
            "It accepts any combination of images in any order with individual labels — ideal for: "
            "(1) applying a specific Ducon product/texture to a user's space; "
            "(2) combining multiple Ducon references into one scene; "
            "(3) using a previous generation as a starting point for refinement; "
            "(4) applying a mood-board or inspiration image alongside a catalog reference. "
            "\n\n"
            "SOURCE CONVENTIONS (passed back to the frontend for resolution):\n"
            "  - Catalog image  → numeric ID string, e.g. '42'\n"
            "  - Catalog by name → image filename/name, e.g. 'marble_pool_coping'\n"
            "  - Previous generation → 'gen:123' where 123 is the generation id\n"
            "  - User upload → upload id string from chat attachment hint, e.g. '7'\n"
            "  - Direct URL → full https:// URL\n"
            "\n\n"
            "LABEL CONVENTIONS (required for quality — same as Studio):\n"
            "  - User's space photo → label MUST be 'User space photo'\n"
            "  - Main Ducon reference → label 'Ducon design direction'\n"
            "  - Extra catalog refs → label 'Ducon product' or descriptive product name\n"
            "  - Put user space FIRST in the images array, then design direction, then products.\n"
            "\n\n"
            "The backend runs the same ImageGenAgent prompt writer + evaluator loop as Studio "
            "when labels match this pattern (enhanced prompt + QC retries).\n"
            "\n\n"
            "MODEL LIMITS: max 10 images total. "
            "Use model='pro' (default) for high quality; model='flash' for faster iterations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Full task description. Reference images by their label and position, e.g. "
                        "'Apply the Ducon marble coping (image 1) to the pool edge in the user space (image 2). "
                        "Match the existing tile colour. Photorealistic, golden-hour lighting.' "
                        "Put specific instructions here — be explicit about what changes and what stays the same."
                    ),
                },
                "images": {
                    "type": "array",
                    "description": (
                        "Ordered list of images. Max 10. Order: user space first, "
                        "Ducon design direction second, then any product refs."
                    ),
                    "maxItems": 10,
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": (
                                    "Image source string: catalog ID ('42'), catalog name, "
                                    "generation ref ('gen:123'), chat upload id ('7'), or URL. "
                                    "For chat attachments use the upload id from the message hint."
                                ),
                            },
                            "label": {
                                "type": "string",
                                "description": (
                                    "Required role label. User photo: 'User space photo'. "
                                    "Main Ducon ref: 'Ducon design direction'. "
                                    "Products: 'Ducon product' or specific product name."
                                ),
                            },
                        },
                        "required": ["source", "label"],
                    },
                },
                "model": {
                    "type": "string",
                    "enum": ["pro", "flash"],
                    "description": (
                        "'pro' (default) = Gemini 3 Pro Image — highest quality, slower. "
                        "'flash' = Gemini 3.1 Flash Image — faster iterations, good quality."
                    ),
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": ["1:1", "4:3", "3:4", "16:9", "9:16"],
                    "description": "Output aspect ratio. Defaults to the model's native ratio if omitted.",
                },
                "show_user": {
                    "type": "boolean",
                    "description": "False for internal agent generations. True only if the user explicitly asked to see generation UI.",
                },
            },
            "required": ["prompt", "images"],
        },
    },
    {
        "type": "function",
        "name": "get_quotation",
        "description": (
            "Requests a Gemini AI quotation analysis for an AI-generated visualisation. "
            "Opens the Quotation modal, lets the user confirm their original space photo "
            "(if not pre-supplied), then returns a full area-measurement and fixed-items breakdown. "
            "Call this after generate_multi_image when the user asks for measurements, a material "
            "list, or a cost estimate. Pass generationRef from the generation result. "
            "If the user already uploaded a photo this session, pass it as userImageRef. "
            "If the user provides any known room dimensions, pass them as referenceMeasurements. "
            "Returns void after opening the quotation modal."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "generationRef": {
                    "type": "string",
                    "description": "Generation reference: id, name, or JSON object string.",
                },
                "userImageRef": {
                    "type": "object",
                    "description": "Optional. User's original space photo upload object.",
                    "properties": {
                        "id":              {"type": "integer"},
                        "name":            {"type": "string"},
                        "filename":        {"type": "string"},
                        "_isUpload":       {"type": "boolean"},
                        "_type":           {"type": "string"},
                    },
                },
                "referenceMeasurements": {
                    "type": "string",
                    "description": (
                        "Optional. Known real-world dimensions, "
                        "e.g. 'Terrace is 8 m wide and 12 m deep. Pool is 4×8 m.'"
                    ),
                },
            },
            "required": ["generationRef"],
        },
    },
]


AUTHENTICATED_ONLY_CHAT_TOOLS = frozenset({"start_designer_job"})


def get_chat_tools(*, user_id: Optional[int] = None) -> list[dict]:
    """Return chat tools available for this request (guests cannot start designer jobs)."""
    if user_id is not None:
        return CHAT_TOOLS
    return [t for t in CHAT_TOOLS if t.get("name") not in AUTHENTICATED_ONLY_CHAT_TOOLS]


# ── File upload helper ────────────────────────────────────────────────────────

async def upload_file_to_gemini(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
) -> dict:
    """
    Upload a file to the Gemini Files API and return its URI and mime_type.

    The Files API URI is temporary (TTL ~48 h) and only usable by the same
    project key.  It is never stored persistently.

    Retries transient upstream errors (503/502/429 and network blips) with
    exponential backoff — Gemini's Files API intermittently returns 503
    Service Unavailable, which is not a caller bug and resolves on retry.

    Returns:
        {"uri": str, "mime_type": str}
    """
    client = get_client()

    # The SDK's async files.upload expects a path or file-like object.
    with tempfile.NamedTemporaryFile(
        suffix=_suffix_from_mime(mime_type), delete=False
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    max_attempts = 4
    delay = 1.5
    last_exc: Exception | None = None
    try:
        for attempt in range(max_attempts):
            try:
                uploaded = await client.aio.files.upload(
                    file=tmp_path,
                    config={"mime_type": mime_type, "display_name": filename},
                )
                return {"uri": uploaded.uri, "mime_type": uploaded.mime_type}
            except Exception as exc:
                last_exc = exc
                if not _is_transient_upload_error(exc) or attempt == max_attempts - 1:
                    raise
                print(
                    f"[ChatAgent] Gemini file upload transient error "
                    f"(attempt {attempt + 1}/{max_attempts}): {exc} — retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10.0)
        # Should be unreachable; surface the last error defensively.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Gemini file upload failed without a specific error.")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _is_transient_upload_error(exc: Exception) -> bool:
    """True for Gemini/HTTP errors that typically resolve on retry."""
    text = str(exc).lower()
    transient_markers = (
        "503", "service unavailable", "502", "bad gateway",
        "429", "rate limit", "resource has been exhausted",
        "504", "gateway timeout", "timeout",
        "server disconnected", "connection reset", "temporarily",
        "unavailable", "try again",
    )
    if any(marker in text for marker in transient_markers):
        return True
    # SDK errors often carry a .status or .code attribute.
    status = getattr(exc, "status", None) or getattr(exc, "code", None)
    if status is not None:
        try:
            code = int(status)
        except (TypeError, ValueError):
            return False
        return code in (429, 502, 503, 504)
    return False


def _suffix_from_mime(mime_type: str) -> str:
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/heic": ".heic",
        "image/heif": ".heif",
        "video/mp4": ".mp4",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "application/pdf": ".pdf",
    }
    return mapping.get(mime_type, ".bin")


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    """Format a dict as a single SSE message."""
    return f"data: {json.dumps(payload)}\n\n"


def _sse_error(message: str) -> str:
    return _sse({"type": "error", "message": message})


# ── Core streaming generator ──────────────────────────────────────────────────

async def stream_chat(
    input_parts: list,
    previous_interaction_id: Optional[str] = None,
    *,
    allow_tools: bool = True,
    user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator that runs one chat turn and yields SSE-formatted strings.

    Handles:
    - Text deltas streamed incrementally
    - Thinking indicators (model reasons via thought_signature — no visible text,
      but we emit a generic indicator so the frontend can show a spinner)
    - Tool call steps collected and emitted as tool_call events
    - Interaction ID tracked and emitted in the final `done` event
    - Error handling: emits an error event and returns cleanly

    Args:
        input_parts:              Gemini input parts — text, image, function_result.
        previous_interaction_id:  Chain to the previous turn for multi-turn history.

    Current SDK event types (google-genai >= 2.0 / Interactions steps schema):
        interaction.created       → event.interaction.id
        interaction.status_update → event.interaction_id
        step.start                → event.step.type ("thought"|"model_output"|"function_call")
        step.delta                → event.delta.type ("text"|"thought_signature"|"arguments_delta")
        step.stop                 → event.index
        interaction.completed     → event.interaction.id, event.interaction.status
    """
    # Route to Claude when enabled — Claude has no server-managed Interactions
    # API, so we keep our own message transcript keyed by the conversation id.
    if llm_provider.use_claude():
        async for chunk in _stream_chat_claude(
            input_parts,
            previous_interaction_id,
            allow_tools=allow_tools,
            user_id=user_id,
        ):
            yield chunk
        return

    client = get_client()

    _chat_model = cfg_str("CHAT_MODEL", CHAT_MODEL)
    _chat_thinking = cfg_str("CHAT_THINKING_LEVEL", CHAT_THINKING_LEVEL)
    _chat_stream = cfg("CHAT_STREAM", CHAT_STREAM)
    chat_tools = get_chat_tools(user_id=user_id) if allow_tools else []

    generation_config: dict = {}
    if _chat_thinking and _chat_thinking.lower() not in ("none", ""):
        generation_config["thinking_level"] = _chat_thinking

    interaction_id: Optional[str] = None
    tool_calls: list[dict] = []
    emitted_tool_call_ids: set[str] = set()

    try:
        _dbg(
            "[CHAT ▶ REQUEST]",
            {
                "model": _chat_model,
                "stream": _chat_stream,
                "thinking_level": _chat_thinking,
                "previous_interaction_id": previous_interaction_id,
                "input_parts": _summarize_input_parts(input_parts),
                "tools": [tool.get("name") for tool in chat_tools],
                "context": (
                    "Interactions API history is server-managed via previous_interaction_id; "
                    "this backend does not manually summarize chat context."
                ),
            },
        )
        if _chat_stream:
            # Track function call steps as they stream in, keyed by step index.
            fc_by_index: dict[int, dict] = {}
            thinking_started = False

            stream = await client.aio.interactions.create(
                model=_chat_model,
                system_instruction=get_chat_system_instruction(),
                input=input_parts,
                tools=chat_tools or None,
                previous_interaction_id=previous_interaction_id,
                generation_config=generation_config or None,
                stream=True,
            )

            async for event in stream:
                etype = getattr(event, "event_type", None)
                usage = _event_usage(event)
                if usage:
                    _dbg("[CHAT ◀ USAGE]", usage)
                    _record_usage(usage, model=_chat_model, user_id=user_id, guest_session_id=guest_session_id)
                else:
                    _dbg("[CHAT ◀ EVENT]", etype, {"index": getattr(event, "index", None)})

                # ── Grab interaction ID as early as possible ────────────────
                if not interaction_id:
                    _ia = getattr(event, "interaction", None)
                    if _ia:
                        interaction_id = getattr(_ia, "id", None)
                    if not interaction_id:
                        interaction_id = getattr(event, "interaction_id", None)

                # ── step.start — track which index is which step type ───────
                if etype == "step.start":
                    step  = getattr(event, "step", None)
                    index = getattr(event, "index", None)
                    stype = getattr(step, "type", None) if step else None

                    if stype == "thought" and not thinking_started:
                        _dbg("[CHAT ◀ THOUGHT]", {"index": index, "note": "thought step started"})
                        # Emit a one-shot indicator so the frontend can show "thinking…"
                        thinking_started = True
                        yield _sse({"type": "thinking_delta", "text": ""})

                    elif stype == "function_call":
                        # Initialise accumulator. The function call metadata is
                        # available on step.start; arguments may arrive as JSON
                        # string chunks in following arguments_delta events.
                        fc_by_index[index] = {
                            "id":        getattr(step, "id", None),
                            "name":      getattr(step, "name", None),
                            "args":      getattr(step, "arguments", {}) or {},
                            "args_text": "",
                        }
                        _dbg("[CHAT ◀ TOOL START]", fc_by_index[index])

                # ── step.delta — text output or streamed function args ──────
                elif etype == "step.delta":
                    delta = getattr(event, "delta", None)
                    index = getattr(event, "index", None)
                    if delta:
                        dtype = getattr(delta, "type", None)

                        if dtype == "text":
                            text = getattr(delta, "text", None)
                            if text:
                                _dbg("[CHAT ◀ TEXT]", repr(text[:240]))
                                yield _sse({"type": "text_delta", "text": text})

                        elif dtype == "arguments_delta":
                            fc = fc_by_index.get(index)
                            if fc is None:
                                fc = {"id": None, "name": None, "args": {}, "args_text": ""}
                                fc_by_index[index] = fc
                            fc["args_text"] += getattr(delta, "arguments", "") or ""
                            _dbg("[CHAT ◀ TOOL ARGS Δ]", {"index": index, "delta": getattr(delta, "arguments", "")})

                # ── step.stop — function call is fully received ─────────────
                elif etype == "step.stop":
                    index = getattr(event, "index", None)
                    fc = fc_by_index.pop(index, None)
                    if fc and fc.get("name"):
                        if fc.get("args_text"):
                            try:
                                fc["args"] = json.loads(fc["args_text"])
                            except json.JSONDecodeError:
                                logger.warning(
                                    "Failed to parse streamed function args for %s: %r",
                                    fc.get("name"), fc.get("args_text"),
                                )
                        fc.pop("args_text", None)
                        call_id = fc.get("id") or f"{fc.get('name')}:{index}"
                        fc["id"] = call_id
                        _dbg("[CHAT ◀ TOOL CALL]", fc)
                        emitted_tool_call_ids.add(str(call_id))
                        yield _sse({
                            "type": "tool_call",
                            "id":   call_id,
                            "name": fc["name"],
                            "args": fc["args"],
                        })

                # ── interaction.completed — final status ────────────────────
                elif etype == "interaction.completed":
                    _ia = getattr(event, "interaction", None)
                    if _ia:
                        interaction_id = getattr(_ia, "id", interaction_id)
                        _dbg(
                            "[CHAT ◀ COMPLETE]",
                            {
                                "interaction_id": interaction_id,
                                "status": getattr(_ia, "status", None),
                                "usage": _interaction_usage(_ia),
                            },
                        )

        else:
            # ── Non-streaming mode ──────────────────────────────────────────
            interaction = await client.aio.interactions.create(
                model=_chat_model,
                system_instruction=get_chat_system_instruction(),
                input=input_parts,
                tools=chat_tools or None,
                previous_interaction_id=previous_interaction_id,
                generation_config=generation_config or None,
            )
            interaction_id = getattr(interaction, "id", None)
            _dbg(
                "[CHAT ◀ RESPONSE]",
                {
                    "interaction_id": interaction_id,
                    "status": getattr(interaction, "status", None),
                    "usage": _interaction_usage(interaction),
                },
            )

            # New Interactions schema returns a typed steps timeline.
            for step in (getattr(interaction, "steps", None) or []):
                stype = getattr(step, "type", None)
                if stype == "model_output":
                    for content in (getattr(step, "content", None) or []):
                        if getattr(content, "type", None) == "text":
                            text = getattr(content, "text", None)
                            if text:
                                yield _sse({"type": "text_delta", "text": text})
                elif stype == "function_call":
                    call_id = getattr(step, "id", None) or f"{getattr(step, 'name', 'tool')}:{len(emitted_tool_call_ids)}"
                    tc = {
                        "id":   call_id,
                        "name": getattr(step, "name", None),
                        "args": getattr(step, "arguments", {}) or {},
                    }
                    _dbg("[CHAT ◀ TOOL CALL]", tc)
                    emitted_tool_call_ids.add(str(call_id))
                    yield _sse({
                        "type": "tool_call",
                        "id":   call_id,
                        "name": tc["name"],
                        "args": tc["args"],
                    })

    except Exception as exc:
        logger.exception("Chat agent error during interaction")
        _dbg("[CHAT ✖ ERROR]", repr(exc))
        await log_error(
            "chat",
            "chat_agent.stream_chat",
            str(exc),
            user_id=user_id,
            guest_session_id=guest_session_id,
            endpoint="/chat/message",
            exc=exc,
        )
        yield _sse_error(str(exc))
        return

    # ── Emit any calls not already emitted in timeline order ──────────────────
    for tc in tool_calls:
        call_id = tc.get("id") or f"{tc.get('name') or 'tool'}:{len(emitted_tool_call_ids)}"
        tc["id"] = call_id
        marker = str(call_id)
        if marker in emitted_tool_call_ids:
            continue
        _dbg("[CHAT ▶ SSE TOOL]", tc)
        yield _sse({
            "type": "tool_call",
            "id":   call_id,
            "name": tc["name"],
            "args": tc["args"],
        })

    # ── Done — always emitted last ────────────────────────────────────────────
    _dbg("[CHAT ▶ DONE]", {"interaction_id": interaction_id})
    yield _sse({"type": "done", "interaction_id": interaction_id})


# ══════════════════════════════════════════════════════════════════════════════
# Claude (Anthropic) chat path — used when USE_CLAUDE is enabled.
#
# Unlike Gemini's Interactions API (server-managed history via
# previous_interaction_id), Anthropic's Messages API is stateless. We keep the
# full conversation transcript in-process, keyed by a stable conversation id.
# That id is returned in the `done` event exactly like an interaction_id, so the
# router/voice/studio/browse memory-sync flows are unchanged: each turn appends
# to the same transcript, and voice/text share it via chat_session.
# ══════════════════════════════════════════════════════════════════════════════

# conversation_id → list[message dict]  (in-process; mirrors chat_session lifetime)
_CLAUDE_HISTORY: dict[str, list[dict]] = {}
_CLAUDE_HISTORY_MAX_MESSAGES = int(cfg("CLAUDE_CHAT_MAX_MESSAGES", 60))

_claude_chat_tools_cache: Optional[list[dict]] = None
_claude_guest_chat_tools_cache: Optional[list[dict]] = None


def get_claude_chat_tools(*, user_id: Optional[int] = None) -> list[dict]:
    global _claude_chat_tools_cache, _claude_guest_chat_tools_cache
    if user_id is not None:
        if _claude_chat_tools_cache is None:
            _claude_chat_tools_cache = llm_provider.to_claude_tools(CHAT_TOOLS)
        return _claude_chat_tools_cache
    if _claude_guest_chat_tools_cache is None:
        _claude_guest_chat_tools_cache = llm_provider.to_claude_tools(
            get_chat_tools(user_id=None),
        )
    return _claude_guest_chat_tools_cache


def _trim_history(messages: list[dict]) -> list[dict]:
    """
    Keep history bounded. Never split a tool_use/tool_result pair and never lead
    with an assistant turn (Anthropic requires the first message to be 'user').
    """
    if len(messages) <= _CLAUDE_HISTORY_MAX_MESSAGES:
        return messages
    trimmed = messages[-_CLAUDE_HISTORY_MAX_MESSAGES:]
    # Drop leading assistant / tool_result messages until we start on a clean user turn.
    while trimmed:
        first = trimmed[0]
        if first.get("role") != "user":
            trimmed = trimmed[1:]
            continue
        blocks = first.get("content")
        if isinstance(blocks, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks
        ):
            trimmed = trimmed[1:]
            continue
        break
    return trimmed


def _last_assistant_tool_use_ids(messages: list[dict]) -> list[dict]:
    """Return [{id,name}] for tool_use blocks of the most recent assistant message."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        ids: list[dict] = []
        for b in msg.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                ids.append({"id": b.get("id"), "name": b.get("name")})
        return ids
    return []


def _build_claude_user_message(input_parts: list, prior_messages: list[dict]) -> dict:
    """
    Convert router input_parts into a single Claude user message.

    Two shapes occur:
      • Normal turn   → text + inline base64 image blocks.
      • Tool results  → function_result parts → tool_result blocks, reconciled
        against the previous assistant message's tool_use ids so Anthropic's
        "every tool_use needs a matching tool_result" rule is always satisfied.
    """
    function_results: dict[str, dict] = {}
    normal_blocks: list[dict] = []

    for part in input_parts:
        if isinstance(part, str):
            if part:
                normal_blocks.append(llm_provider.text_block(part))
            continue
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            txt = part.get("text") or ""
            if txt:
                normal_blocks.append(llm_provider.text_block(txt))
        elif ptype == "image_b64":
            data_b64 = part.get("data")
            if data_b64:
                normal_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.get("mime_type") or "image/jpeg",
                        "data": data_b64,
                    },
                })
        elif ptype == "image_url":
            url = part.get("url")
            if url:
                normal_blocks.append({
                    "type": "image",
                    "source": {"type": "url", "url": url},
                })
        elif ptype == "document_b64":
            data_b64 = part.get("data")
            if data_b64:
                normal_blocks.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": part.get("mime_type") or "application/pdf",
                        "data": data_b64,
                    },
                })
        elif ptype == "function_result":
            call_id = part.get("call_id") or part.get("name")
            text = _function_result_to_text(part.get("result"))
            function_results[str(call_id)] = {
                "text": text,
                "is_error": _function_result_is_error(part.get("result")),
            }

    if function_results:
        expected = _last_assistant_tool_use_ids(prior_messages)
        blocks: list[dict] = []
        used: set[str] = set()
        for tu in expected:
            cid = str(tu.get("id"))
            res = function_results.get(cid)
            used.add(cid)
            if res is None:
                blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tu.get("id"),
                    "content": [{"type": "text", "text": "No result was returned for this tool call."}],
                    "is_error": True,
                })
            else:
                blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tu.get("id"),
                    "content": [{"type": "text", "text": res["text"]}],
                    "is_error": res["is_error"],
                })
        # Include any extra results that didn't match a known tool_use (defensive).
        for cid, res in function_results.items():
            if cid in used:
                continue
            blocks.append({
                "type": "tool_result",
                "tool_use_id": cid,
                "content": [{"type": "text", "text": res["text"]}],
                "is_error": res["is_error"],
            })
        return {"role": "user", "content": blocks}

    if not normal_blocks:
        normal_blocks = [llm_provider.text_block("(no content)")]
    return {"role": "user", "content": normal_blocks}


def _function_result_to_text(result: object) -> str:
    """The router wraps tool results as [{'type':'text','text':...}]; unwrap to text."""
    if isinstance(result, list):
        texts = []
        for item in result:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text") or "")
            else:
                texts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(t for t in texts if t)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


def _function_result_is_error(result: object) -> bool:
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            txt = (first.get("text") or "")
            return txt.startswith("Error:")
    return False


async def _stream_chat_claude(
    input_parts: list,
    previous_interaction_id: Optional[str],
    *,
    allow_tools: bool = True,
    user_id: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """Claude equivalent of stream_chat — same SSE event contract."""
    conv_id = previous_interaction_id if (previous_interaction_id and str(previous_interaction_id).startswith("cld_")) else f"cld_{uuid.uuid4().hex}"
    messages: list[dict] = list(_CLAUDE_HISTORY.get(conv_id, []))
    if previous_interaction_id and str(previous_interaction_id).startswith("cld_") and not messages:
        # KNOWN LIMITATION: _CLAUDE_HISTORY is per-worker in-memory. Under
        # gunicorn -w N a continuation can land on a worker that never saw the
        # conversation — context is lost and tool_result pairing can 400.
        # Requires sticky sessions or -w 1 while USE_CLAUDE is enabled.
        logger.warning(
            "[CHAT · claude] continuation %s has no in-memory history on this "
            "worker (multi-worker without sticky sessions?) — starting fresh",
            conv_id,
        )

    user_msg = _build_claude_user_message(input_parts, messages)
    messages.append(user_msg)

    _dbg("[CHAT ▶ REQUEST · claude]", {
        "model": llm_provider.CLAUDE_MODEL,
        "conv_id": conv_id,
        "history_messages": len(messages),
        "allow_tools": allow_tools,
        "input_parts": _summarize_input_parts(input_parts),
    })

    try:
        client = llm_provider.get_async_anthropic_client()
        kwargs = {
            "model": llm_provider.CLAUDE_MODEL,
            "max_tokens": llm_provider.CLAUDE_MAX_TOKENS,
            "system": get_chat_system_instruction(),
            "messages": messages,
        }
        if allow_tools:
            kwargs["tools"] = get_claude_chat_tools(user_id=user_id)
        th = llm_provider._thinking_param()
        if th:
            kwargs["thinking"] = th

        thinking_started = False
        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    dtype = getattr(delta, "type", None) if delta else None
                    if dtype == "thinking_delta":
                        if not thinking_started:
                            thinking_started = True
                            yield _sse({"type": "thinking_delta", "text": ""})
                        chunk = getattr(delta, "thinking", "") or ""
                        if chunk:
                            yield _sse({"type": "thinking_delta", "text": chunk})
                    elif dtype == "text_delta":
                        chunk = getattr(delta, "text", "") or ""
                        if chunk:
                            yield _sse({"type": "text_delta", "text": chunk})
            final = await stream.get_final_message()

        # Persist the assistant turn (preserving thinking + tool_use blocks).
        messages.append({"role": "assistant", "content": llm_provider.serialize_content(final)})
        _CLAUDE_HISTORY[conv_id] = _trim_history(messages)

        # Emit any tool calls the model requested.
        for tu in llm_provider.tool_use_blocks(final):
            call_id = tu.get("id") or f"{tu.get('name') or 'tool'}:{uuid.uuid4().hex[:8]}"
            _dbg("[CHAT ◀ TOOL CALL · claude]", {"id": call_id, "name": tu.get("name")})
            yield _sse({
                "type": "tool_call",
                "id":   call_id,
                "name": tu.get("name"),
                "args": tu.get("input") or {},
            })

    except Exception as exc:
        logger.exception("Claude chat agent error")
        _dbg("[CHAT ✖ ERROR · claude]", repr(exc))
        await log_error(
            "chat",
            "chat_agent.stream_chat_claude",
            str(exc),
            endpoint="/chat/message",
            exc=exc,
        )
        yield _sse_error(str(exc))
        return

    _dbg("[CHAT ▶ DONE · claude]", {"interaction_id": conv_id})
    yield _sse({"type": "done", "interaction_id": conv_id})
