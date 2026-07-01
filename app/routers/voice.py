"""
Real-time conversational AI — WebSocket bridge router.

Endpoint
--------
  GET /ws/voice

  Query parameters:
    token  (optional) — JWT access token for authenticated users.
           Pass as ?token=<jwt> because browsers cannot set the
           Authorization header on WebSocket connections.

  Session lifetime
  ----------------
    Each WebSocket connection maps to exactly one fresh Gemini Live session.
    Text-chat history (when the user has an active chat session) is seeded
    into the Live session so voice can continue where text left off.
    Within a single connection, the backend transparently handles:
    • The Gemini ~10-minute WSS connection limit (reconnects with a
      server-issued resumption handle, context preserved).
    • Transient keepalive drops (reconnect with exponential back-off).

  Session resumption handles are kept entirely server-side and are never
  sent to the client.  This guarantees history isolation between sessions.

Inbound messages from the client (JSON)
----------------------------------------
  { "type": "audio_chunk",      "data": "<base64 PCM 16-bit 16 kHz>" }
  { "type": "text",             "data": "<user text message>" }
  { "type": "audio_stream_end" }   # microphone paused / muted
  { "type": "ping"             }   # keep-alive (backend responds with pong)
  { "type": "tool_pause",
    "call_id": "<id from tool_call>",
    "name":    "<function name>" }  # sent immediately for human-input tools (e.g. UploadImage)
                                    # cancels the 30s auto-timeout; tool_result still follows
  { "type": "tool_result",
    "call_id": "<id from tool_call>",
    "name":    "<function name>",
    "result":  { ... },             # JSON result from window.__duconAPI.<name>()
    "error":   "<msg>"   }          # omit result and include error on failure

Outbound messages to the client (JSON)
----------------------------------------
  { "type": "connected"        }   # Gemini session ready (also sent on internal reconnect)
  { "type": "reconnecting",     "message": "Reconnecting in 1s…" }
  { "type": "audio_chunk",      "data": "<base64 PCM 24 kHz>",
                                 "mime_type": "audio/pcm;rate=24000" }
  { "type": "input_transcript", "data": "<what Gemini heard>" }
  { "type": "output_transcript","data": "<what Gemini is saying>" }
  { "type": "turn_complete"    }
  { "type": "interrupted"      }   # user barged in; flush your audio queue
  { "type": "go_away",          "time_left_ms": 60000 }   # informational
  { "type": "tool_call",
    "call_id": "<opaque id>",
    "name":    "<function name>",
    "args":    { ... }  }           # see ★ Tool call protocol below
  { "type": "error",            "message": "<description>" }
  { "type": "pong"             }

★ Tool call protocol (frontend must follow this exactly)
---------------------------------------------------------
Audio ordering explained
~~~~~~~~~~~~~~~~~~~~~~~~
The model's narration audio (PCM chunks) and the tool_call event are SEPARATE
WebSocket messages from the backend.  The audio always arrives first; the
tool_call event arrives after the last audio chunk for that narration.  The
output_transcript events you may see arriving after tool_call are just the
delayed text transcription of audio that was already sent — they are for
captions only and do not affect ordering.

Correct frontend behaviour on receiving a "tool_call" event
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run audio playback and tool execution CONCURRENTLY, not sequentially:

  1. START PLAYING THE AUDIO BUFFER immediately (the narration audio is
     already queued — let it play through without blocking).
  2. START EXECUTING the tool at the same time.
     Do NOT pause or hold the audio queue while the tool runs.
     Do NOT wait for the tool to finish before playing audio.

  Exception — UI-opening tools (UploadImage, show_*):
     Wait for the audio buffer to drain *before* opening the modal/panel,
     so the user hears "I'm opening the upload window" before it appears.
     Execute the tool immediately after audio drains (do not delay further).

  Summary by tool:
    get_selected_image, AISearch, KeywordSearch, get_image
        → execute immediately, audio plays in parallel.
    show_user_uploaded_images, show_image_generations, show_bookmarks
        → wait for audio drain, then open panel, send tool_result.
    UploadImage
        → wait for audio drain, open file picker, send tool_pause to
          backend (cancels the 30 s auto-timeout), send tool_result
          when user completes upload.
    generate_multi_image
        → execute immediately (no tool_pause needed, backend has no
          timeout for this tool), send tool_result when Promise resolves.

  On error: send tool_result with "error" set and no "result" field.

Error codes
-----------
  4001  Token provided but invalid or expired
  4003  Internal error starting Gemini session
"""

from __future__ import annotations

import asyncio
from app.hashing import sha256_hex
import json
import logging
import uuid
from typing import Awaitable, Callable, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.models import User
from app.live_session import GeminiLiveSession, LiveEvent, LiveEventType, _dbg
from app.auth import resolve_user_from_token
from app import chat_session
from app.guest_usage import (
    GuestUsageKind,
    enforce_guest_limit,
    increment_guest_usage,
)
from app.routers.guest import get_or_create_guest_session

router = APIRouter(tags=["voice"])
logger = logging.getLogger(__name__)


def _hash_ip(ip: str) -> str:
    return sha256_hex(ip)


# ── Auth helper ───────────────────────────────────────────────────────────────

async def _resolve_user(token: Optional[str], db: AsyncSession) -> Optional[User]:
    """Decode an optional JWT and return the User row (or None for guests)."""
    return await resolve_user_from_token(token, db)


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@router.websocket("/ws/voice")
async def voice_ws(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None, description="JWT access token"),
    guest_session_id: Optional[str] = Query(
        default=None,
        description="Guest session UUID (required for unauthenticated voice)",
    ),
):
    """
    Bidirectional WebSocket bridge between the client and Gemini Live API.

    Every connection starts a fresh Gemini session — no history is carried
    over from previous conversations.  Session resumption handles are managed
    internally by the backend to survive the ~10-minute WSS limit.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    client_ip = websocket.client.host if websocket.client else "unknown"

    # ── Resolve user or guest session ─────────────────────────────────────────
    user: Optional[User] = None
    guest_db_session_id: Optional[str] = None

    if token:
        async for db in get_db():
            user = await _resolve_user(token, db)
            if user is None:
                await websocket.close(code=4001, reason="Invalid or expired token")
                logger.warning("Voice WS rejected — invalid token (session %s)", session_id)
                return
            break

    elif guest_session_id:
        guest_db_session_id = guest_session_id
        try:
            async for db in get_db():
                await get_or_create_guest_session(
                    db,
                    guest_session_id,
                    client_ip,
                    usage_kind=GuestUsageKind.VOICE,
                )
                await db.commit()
                break
        except Exception as exc:
            from fastapi import HTTPException
            if isinstance(exc, HTTPException) and exc.status_code == 429:
                await _send_json(
                    websocket,
                    {
                        "type": "error",
                        "message": "Guest limit reached. Sign up to continue.",
                        "code": "GUEST_LIMIT_REACHED",
                    },
                )
                await websocket.close(code=4003, reason="Guest limit reached")
                return
            raise
    else:
        await websocket.close(code=4001, reason="Authentication required")
        logger.warning("Voice WS rejected — no token or guest_session_id (session %s)", session_id)
        return

    user_id = user.id if user else None
    user_label = f"user={user_id}" if user_id else f"guest={guest_db_session_id}"
    logger.info("Voice WS connected — %s session=%s", user_label, session_id)

    async def _increment_guest_voice() -> bool:
        """Count one completed voice turn; return False when guest limit is exhausted."""
        if not guest_db_session_id:
            return True
        try:
            async for db in get_db():
                row = await enforce_guest_limit(
                    db,
                    guest_db_session_id,
                    _hash_ip(client_ip),
                    GuestUsageKind.VOICE,
                )
                await increment_guest_usage(db, row, GuestUsageKind.VOICE)
                await db.commit()
                return True
        except Exception as exc:
            from fastapi import HTTPException
            if isinstance(exc, HTTPException) and exc.status_code == 429:
                return False
            logger.exception("Voice WS guest usage increment failed — session=%s", session_id)
            return True

    # ── Create Gemini Live session — seed chat transcript when available ───────
    gemini_session = GeminiLiveSession(
        user_id=user_id,
        guest_session_id=guest_db_session_id,
    )
    if user_id is not None:
        seed_turns = chat_session.get_voice_seed_turns(user_id)
        if seed_turns:
            gemini_session.seed_history(seed_turns)
            logger.info(
                "Voice WS seeding %d chat turn(s) — user=%s session=%s",
                len(seed_turns), user_id, session_id,
            )
    elif guest_db_session_id:
        seed_turns = chat_session.get_guest_voice_seed_turns(guest_db_session_id)
        if seed_turns:
            gemini_session.seed_history(seed_turns)
            logger.info(
                "Voice WS seeding %d chat turn(s) — guest=%s session=%s",
                len(seed_turns), guest_db_session_id, session_id,
            )
    event_queue: asyncio.Queue[LiveEvent] = asyncio.Queue()

    try:
        await gemini_session.start(event_queue)
    except Exception as exc:
        logger.exception("Failed to start Gemini Live session — %s", user_label)
        await _send_json(websocket, {"type": "error", "message": str(exc)})
        await websocket.close(code=4003, reason="Gemini session start failed")
        return

    # ── Run both relay tasks concurrently ─────────────────────────────────────
    # Note: the initial "connected" event is delivered via the queue relay below,
    # keeping a single code path for both the first connect and reconnects.
    client_to_gemini_task = asyncio.create_task(
        _relay_client_to_gemini(websocket, gemini_session, session_id),
        name=f"client→gemini-{session_id[:8]}",
    )
    gemini_to_client_task = asyncio.create_task(
        _relay_gemini_to_client(
            websocket,
            event_queue,
            session_id,
            on_turn_complete=_increment_guest_voice if guest_db_session_id else None,
        ),
        name=f"gemini→client-{session_id[:8]}",
    )

    try:
        # Wait until either direction finishes (disconnect, error, or GoAway)
        done, pending = await asyncio.wait(
            {client_to_gemini_task, gemini_to_client_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
    finally:
        await gemini_session.close()
        logger.info("Voice WS disconnected — %s session=%s", user_label, session_id)


# ── Relay: client → Gemini ────────────────────────────────────────────────────

async def _relay_client_to_gemini(
    ws: WebSocket,
    session: GeminiLiveSession,
    session_id: str,
) -> None:
    """
    Read JSON messages from the client WebSocket and dispatch to the Gemini
    session.  Returns (and causes the parent task to cancel the sibling) when
    the WebSocket disconnects or receives an unrecognised / malformed message.
    """
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Voice WS non-JSON message ignored — session=%s", session_id)
                continue

            msg_type = msg.get("type", "")

            if msg_type == "audio_chunk":
                data = msg.get("data")
                if data:
                    mime = msg.get("mime_type", "audio/pcm;rate=16000")
                    logger.debug(
                        "→ Gemini audio chunk len=%d — session=%s",
                        len(data), session_id,
                    )
                    await session.send_audio(data, mime_type=mime)

            elif msg_type == "text":
                data = msg.get("data", "").strip()
                if data:
                    logger.info(
                        "→ Gemini text: %r — session=%s", data[:80], session_id
                    )
                    _dbg(f"[FRONT ▶ TEXT ] \"{data[:120]}\"")
                    await session.send_text(data)

            elif msg_type == "audio_stream_end":
                logger.info("→ Gemini audio_stream_end — session=%s", session_id)
                _dbg(f"[FRONT ▶ END  ] audio_stream_end")

                await session.send_audio_stream_end()

            elif msg_type == "tool_pause":
                call_id = msg.get("call_id", "")
                name    = msg.get("name", "")
                logger.info(
                    "← Frontend tool_pause: %s call_id=%s (timeout cancelled) — session=%s",
                    name, call_id, session_id,
                )
                _dbg(f"[FRONT ▶ PAUSE] {name}  call_id={call_id}")
                session.cancel_tool_timeout(call_id)

            elif msg_type == "tool_result":
                call_id = msg.get("call_id", "")
                name    = msg.get("name", "")
                result  = msg.get("result")   # may be None if error
                error   = msg.get("error")    # str or None
                logger.info(
                    "← Frontend tool_result: %s call_id=%s error=%s — session=%s",
                    name, call_id, bool(error), session_id,
                )
                # Show raw result keys + any id/name values so we can trace
                # whether critical IDs (needed for generation tools) are present.
                if result is not None:
                    if isinstance(result, dict):
                        id_val   = result.get("id")
                        name_val = result.get("name") or result.get("filename")
                        keys     = list(result.keys())
                        _dbg(f"[FRONT ▶ TRES ] {name}  error={bool(error)}"
                             f"  id={id_val}  name={name_val!r}  keys={keys}")
                    elif isinstance(result, list):
                        _dbg(f"[FRONT ▶ TRES ] {name}  error={bool(error)}"
                             f"  list[{len(result)}] items")
                    else:
                        _dbg(f"[FRONT ▶ TRES ] {name}  error={bool(error)}"
                             f"  result={repr(result)[:80]}")
                else:
                    _dbg(f"[FRONT ▶ TRES ] {name}  error={error!r}  result=None")
                await session.submit_tool_result(
                    call_id=call_id,
                    name=name,
                    result=result,
                    error=error,
                )

            elif msg_type == "ping":
                await _send_json(ws, {"type": "pong"})

            else:
                logger.debug(
                    "Voice WS unknown message type %r — session=%s", msg_type, session_id
                )

    except WebSocketDisconnect:
        logger.info("Voice WS client disconnected — session=%s", session_id)
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Voice WS client relay error — session=%s", session_id)


# ── Relay: Gemini → client ────────────────────────────────────────────────────

async def _relay_gemini_to_client(
    ws: WebSocket,
    event_queue: asyncio.Queue[LiveEvent],
    session_id: str,
    on_turn_complete: Optional[Callable[[], Awaitable[bool]]] = None,
) -> None:
    """
    Drain the LiveEvent queue and forward each event as JSON to the client.
    Returns when the WebSocket is no longer writable or an error event arrives.
    """
    try:
        while True:
            event: LiveEvent = await event_queue.get()
            payload = event.to_dict()

            # Log tool calls explicitly so we can confirm the event reaches
            # the relay and is about to be pushed to the frontend.
            if event.type == LiveEventType.TOOL_CALL:
                logger.info(
                    "→ Frontend TOOL_CALL: %s call_id=%s args=%s — session=%s",
                    event.name, event.call_id, list((event.args or {}).keys()),
                    session_id,
                )

            try:
                await _send_json(ws, payload)
            except (WebSocketDisconnect, RuntimeError) as exc:
                logger.info(
                    "Voice WS write failed (client gone): %s — session=%s",
                    exc, session_id,
                )
                return
            except Exception:
                logger.exception(
                    "Voice WS send error (event_type=%s) — session=%s",
                    payload.get("type"), session_id,
                )
                continue

            if event.type == LiveEventType.TURN_COMPLETE and on_turn_complete:
                try:
                    allowed = await on_turn_complete()
                except Exception:
                    logger.exception(
                        "Voice WS guest usage increment failed — session=%s",
                        session_id,
                    )
                    allowed = True
                if not allowed:
                    await _send_json(
                        ws,
                        {
                            "type": "error",
                            "message": "Guest limit reached. Sign up to continue.",
                            "code": "GUEST_LIMIT_REACHED",
                        },
                    )
                    await ws.close(code=4003, reason="Guest limit reached")
                    return

            if event.type == LiveEventType.ERROR:
                await ws.close(code=1011, reason=event.message or "Gemini error")
                return

    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Voice WS Gemini relay error — session=%s", session_id)


# ── Utility ────────────────────────────────────────────────────────────────────

async def _send_json(ws: WebSocket, payload: dict) -> None:
    """Serialise payload and send as a text frame."""
    await ws.send_text(json.dumps(payload, ensure_ascii=False))
