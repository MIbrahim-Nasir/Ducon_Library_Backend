"""
Gemini Live API session manager.

One GeminiLiveSession is created per connected WebSocket client.  It bridges
between the client (audio/text in, audio/transcripts out) and the Gemini
Live API using the google-genai async SDK.

Architecture
------------
  Client WebSocket
        │  audio_chunk / text / audio_stream_end / tool_result
        ▼
  GeminiLiveSession.send_*() / submit_tool_result()
        │
        ▼  google-genai SDK (WSS)
  Gemini Live API
        │  audio, transcripts, interrupted, turn_complete, go_away, handle, tool_call
        ▼
  GeminiLiveSession._session_runner()  ──push──►  asyncio.Queue[LiveEvent]
        │
        ▼
  voice.py router reads queue and forwards over client WebSocket

Tool-call bridge
----------------
  When Gemini decides to call a tool it sends a tool_call message.
  The backend pushes a TOOL_CALL LiveEvent to the queue; voice.py
  serialises it and sends it to the frontend.  The frontend calls
  window.__duconAPI.<name>(args) and sends back a tool_result message.
  voice.py passes it to submit_tool_result(), which calls
  session.send_tool_response() so Gemini can continue its turn.

Reliability features
--------------------
  • _session_runner owns a single `async with client.aio.live.connect()`
    block.  This is the only correct way to use the SDK — it ensures
    the SDK can manage its own WebSocket keepalive and teardown.

  • If the underlying WSS connection drops (keepalive timeout, server
    restart, GoAway, etc.) _session_runner catches the exception and
    transparently reconnects using the latest resumption handle.
    Exponential back-off prevents hammering the API on repeated failures.

  • send_* methods never drop messages silently.  If the session is in
    the middle of a reconnect, they wait up to SEND_WAIT_TIMEOUT seconds
    for the connection to be restored before giving up.

  • close() simply cancels the runner task; the `async with` block then
    exits cleanly, avoiding the TimeoutError that manual __aexit__ calls
    can produce.

  • Conversation history is accumulated from input/output transcripts and
    re-seeded into every new Gemini session.  If the underlying WSS closes
    between turns (code 1000) and a fresh session is started, Gemini still
    has the full transcript of everything said so far, so it never greets
    again or forgets earlier context.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Model ─────────────────────────────────────────────────────────────────────
from app.admin.settings_store import cfg

LIVE_MODEL = "gemini-3.1-flash-live-preview"        # default; live via cfg("LIVE_MODEL", LIVE_MODEL)

# ── Voice ─────────────────────────────────────────────────────────────────────
LIVE_VOICE = "Kore"

# ── Thinking level ────────────────────────────────────────────────────────────
# gemini-3.1-flash-live uses thinkingLevel: minimal | low | medium | high
# "minimal" keeps latency lowest (default).
LIVE_THINKING_LEVEL = "minimal"

# ── Context-window compression ────────────────────────────────────────────────
_COMPRESSION_TRIGGER_TOKENS = 100000
_COMPRESSION_TARGET_TOKENS = 32000

# ── Default system instruction ────────────────────────────────────────────────
from app import prompt_loader


def get_live_system_instruction() -> str:
    override = os.getenv("LIVE_SYSTEM_INSTRUCTION")
    if override:
        return override.strip()
    prompt_loader.ensure_prompts_loaded()
    return prompt_loader.LIVE_VOICE_AGENT_SYSTEM


# ── Debug printing ─────────────────────────────────────────────────────────────
# Set LIVE_DEBUG=true in .env (or via admin panel) to enable verbose terminal prints
# of every message sent to / received from Gemini Live. Leave empty or false for production.
_LIVE_DEBUG: bool = False


def _dbg(*args) -> None:
    """Print only when LIVE_DEBUG is enabled."""
    if cfg("LIVE_DEBUG", _LIVE_DEBUG):
        print(*args)


from app.search_tools import ai_search_live_declaration, keyword_search_live_declaration

# ── Tool declarations ──────────────────────────────────────────────────────────
# All tools execute on the frontend via window.__duconAPI.*.
_TOOLS: list[dict] = [
    {
        "function_declarations": [
            ai_search_live_declaration(),
            keyword_search_live_declaration(require_query=False),
            {
                "name": "UploadImage",
                "description": (
                    "Opens the file picker so the user can upload a photo of their space. "
                    "Resolves with {id, name, filename, _isUpload:true, _type:'user_upload'} once the "
                    "user confirms. Pass id or the full object as a user upload source in "
                    "generate_multi_image. Use when no upload_id is available from a chat attachment."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "get_selected_image",
                "description": (
                    "Returns the catalog image currently open in the Image Viewer, or null. "
                    "Call this proactively whenever the user refers to 'this image', 'this design', "
                    "'something like this', or any phrasing that implies they are looking at something. "
                    "Do not ask the user which image they mean until you have tried this first."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "get_image",
                "description": (
                    "Open an image in the Image Viewer and return it. Resolves catalog images, AI "
                    "generations, and user uploads in priority order. Use after AISearch when you need "
                    "proper visual understanding or a concrete design/product reference. Returns "
                    "CatalogImage | UploadImage | AIGeneration."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ref": {
                            "type": "string",
                            "description": "Image reference: numeric ID, name/filename, generation ref, upload ref, or JSON object with id.",
                        }
                    },
                    "required": ["ref"],
                },
            },
            {
                "name": "show_user_uploaded_images",
                "description": "Open the Uploads panel in the sidebar showing the user's previously uploaded photos.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "show_image_generations",
                "description": "Open the AI Generations panel in the sidebar showing completed design previews.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "show_bookmarks",
                "description": "Open the Bookmarks panel in the sidebar showing items the user has saved.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "open_sidebar",
                "description": (
                    "Open the sidebar workspace panel. Returns {status:'open'}. "
                    "Use when the user asks to open the sidebar or wants to access their workspace."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "open_ai_generations",
                "description": (
                    "Open the AI Generations panel in the sidebar and return the list of the user's "
                    "AI-generated designs. Returns Array<{id, generation_name, url, ducon_image_id}>. "
                    "Use when the user asks to see, browse, or access their generated images."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "open_catalog",
                "description": (
                    "Navigate to the main catalog view and scroll so the category tabs and design grid "
                    "are visible at the top. Returns {status:'open'}. "
                    "Use when the user wants to browse or explore the full catalog."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "get_quotation",
                "description": (
                    "Requests a Gemini AI quotation analysis for an AI-generated visualisation. "
                    "Opens the Quotation modal, lets the user confirm their original space photo "
                    "(if not pre-supplied), then returns a full area-measurement and fixed-items breakdown. "
                    "Call this after generate_multi_image when the user asks for measurements, a material "
                    "list, or a cost estimate. Pass generationRef from the generation result. "
                    "If the user already uploaded a photo this session, pass it as userImageRef to skip "
                    "the before-photo picker. If the user provides any known room or space dimensions, "
                    "pass them as referenceMeasurements — this raises estimate accuracy significantly. "
                    "Returns void after opening the quotation modal."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "generationRef": {
                            "type": "string",
                            "description": (
                                "Generation reference: id, name, or JSON object string."
                            ),
                        },
                        "userImageRef": {
                            "type": "object",
                            "description": (
                                "Optional. The user's original space photo upload object."
                            ),
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
                                "Optional. Known real-world dimensions that help Gemini calibrate area "
                                "estimates, e.g. 'Terrace is 8 m wide and 12 m deep. Pool is 4×8 m.' "
                                "Omit or pass null when dimensions are unknown. When provided, confidence "
                                "levels in the response will be higher."
                            ),
                        },
                    },
                    "required": ["generationRef"],
                },
            },
            {
                "name": "send_to_chat",
                "description": (
                    "Delegate a design task to the shared text chat agent (same session as the "
                    "chat panel). Use for autonomous designer jobs and any multi-step design "
                    "workflow — especially start_designer_job. The chat agent executes tools, "
                    "streams progress in chat, and returns a summary. Do not call start_designer_job "
                    "directly from voice; always use send_to_chat for designer jobs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": (
                                "Clear instruction for the chat agent, e.g. "
                                "'Run start_designer_job for the user's terrace — they want a modern Ducon look.'"
                            ),
                        },
                        "user_upload_image": {
                            "type": "string",
                            "description": (
                                "Upload id/source for the client's space image (upload:N or id from "
                                "UploadImage). Include in message if omitted here."
                            ),
                        },
                    },
                    "required": ["message"],
                },
            },
            {
                "name": "generate_multi_image",
                "description": (
                    "Generate a new design image by compositing multiple source images with a custom prompt. "
                    "Use this for both simple single-reference generations and advanced multi-image tasks. "
                    "It accepts any combination of images in any order with individual labels. "
                    "Use it to apply Ducon products/textures to a user's space, combine multiple Ducon "
                    "references, refine a previous generation, or use inspiration/mood-board images. "
                    "Sources can be catalog IDs, catalog names/filenames, generation refs like 'gen:123', "
                    "upload IDs from UploadImage or chat attachments, 'upload' to open the picker, or URLs. "
                    "Max 10 images. Returns {id/generation_id, generation_name, signed_url, ...}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": (
                                "Full task description. Reference images by their label and position, e.g. "
                                "'Apply the Ducon marble coping (image 1) to the pool edge in the user space "
                                "(image 2). Preserve existing architecture. Photorealistic.'"
                            ),
                        },
                        "images": {
                            "type": "array",
                            "description": "Ordered source images. Max 10. Order matters.",
                            "maxItems": 10,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {
                                        "type": "string",
                                        "description": (
                                            "Catalog ID/name, 'gen:123', upload id, 'upload', or URL."
                                        ),
                                    },
                                    "label": {
                                        "type": "string",
                                        "description": "Short image label used in the prompt, e.g. 'user terrace'.",
                                    },
                                },
                                "required": ["source", "label"],
                            },
                        },
                        "model": {
                            "type": "string",
                            "enum": ["pro", "flash"],
                            "description": "'pro' for highest quality, 'flash' for faster iterations.",
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "enum": ["1:1", "4:3", "3:4", "16:9", "9:16"],
                            "description": "Output aspect ratio. Omit for default.",
                        },
                    },
                    "required": ["prompt", "images"],
                },
            },
        ]
    }
]

# ── Reconnect policy ──────────────────────────────────────────────────────────
_RECONNECT_INITIAL_DELAY = 1.0   # seconds before first retry
_RECONNECT_MAX_DELAY     = 30.0  # cap on exponential backoff

# ── Tool call timeout ─────────────────────────────────────────────────────────
# Gemini 3.1 Flash Live uses synchronous-only function calling: the model
# pauses and waits for a FunctionResponse before continuing.  If the frontend
# never sends a tool_result, the receive() loop blocks forever and the agent
# appears frozen.  This timeout auto-responds with an error so Gemini can
# continue rather than waiting indefinitely.
_TOOL_CALL_TIMEOUT_SECS = 30.0  # default; live via cfg("LIVE_TOOL_CALL_TIMEOUT", _TOOL_CALL_TIMEOUT_SECS)

# Tools that must NOT be auto-timed-out because they are either:
#   • user-paced   — the user takes as long as they need (UploadImage)
#   • long-running — the backend operation itself can take > 30 s (generate_multi_image)
# These tools rely solely on the frontend sending a tool_result when done.
# The frontend must still send tool_pause for UploadImage to signal that the
# user interaction has started, but the auto-timeout is never created for any
# tool in this set.
_NO_TIMEOUT_TOOLS: frozenset[str] = frozenset({
    "UploadImage",
    "generate_multi_image",
    "send_to_chat",
    "get_quotation",   # Gemini 3.1 Pro analysis can take 15–30 s
})

_LIVE_AISEARCH_MAX_ITEMS = 8        # default; live via cfg("CHAT_AISEARCH_MAX_ITEMS", ...)
_LIVE_TOOL_RESULT_MAX_CHARS = 12000


def _live_aisearch_max_items() -> int:
    return int(cfg("LIVE_AISEARCH_MAX_ITEMS", cfg("CHAT_AISEARCH_MAX_ITEMS", _LIVE_AISEARCH_MAX_ITEMS)))


def _live_tool_result_max_chars() -> int:
    return int(cfg("LIVE_TOOL_RESULT_MAX_CHARS", cfg("CHAT_TOOL_RESULT_MAX_CHARS", _LIVE_TOOL_RESULT_MAX_CHARS)))


def _tool_call_timeout() -> float:
    return float(cfg("LIVE_TOOL_CALL_TIMEOUT", _TOOL_CALL_TIMEOUT_SECS))

# ── Conversation history ───────────────────────────────────────────────────────
# Maximum number of turns kept in-memory for re-seeding on reconnect.
# Each "turn" is one Content dict (role=user or role=model).
# 40 turns = 20 user messages + 20 model replies before the oldest are pruned.
_MAX_HISTORY_TURNS: int = 40

# ── Missed-tool-call recovery ──────────────────────────────────────────────────
# Maximum number of consecutive [execute_now] triggers sent for the same
# unexecuted action.  Prevents infinite retry loops if the model keeps
# ignoring the prompt.
_MAX_EXECUTE_NOW_RETRIES: int = 2

# ── Send-wait policy ──────────────────────────────────────────────────────────
# How long send_* methods will wait for the session to (re)connect before
# giving up on a single message.
SEND_WAIT_TIMEOUT = 8.0  # default; live via cfg("LIVE_SEND_WAIT_TIMEOUT", SEND_WAIT_TIMEOUT)


def _send_wait_timeout() -> float:
    return float(cfg("LIVE_SEND_WAIT_TIMEOUT", SEND_WAIT_TIMEOUT))


# ── VAD sensitivity ───────────────────────────────────────────────────────────
# How many milliseconds of silence signal end-of-speech.  Tweak via admin panel
# or .env if the model cuts off too early (lower value) or too late (raise value).
_VAD_SILENCE_MS = 800
# prefix_padding: audio captured before speech start is included to avoid
# clipping the first syllable.
_VAD_PREFIX_MS = 100


def _vad_silence_ms() -> int:
    return int(cfg("LIVE_VAD_SILENCE_MS", _VAD_SILENCE_MS))


def _vad_prefix_ms() -> int:
    return int(cfg("LIVE_VAD_PREFIX_MS", _VAD_PREFIX_MS))


# ── Event types ────────────────────────────────────────────────────────────────
class LiveEventType(str, Enum):
    CONNECTED         = "connected"
    RECONNECTING      = "reconnecting"   # backend reconnecting to Gemini, transparent to user
    AUDIO_CHUNK       = "audio_chunk"
    INPUT_TRANSCRIPT  = "input_transcript"
    OUTPUT_TRANSCRIPT = "output_transcript"
    TURN_COMPLETE     = "turn_complete"
    INTERRUPTED       = "interrupted"    # user barge-in; flush audio queue
    GO_AWAY           = "go_away"        # Gemini WSS closing soon (informational)
    TOOL_CALL         = "tool_call"      # Gemini wants to invoke a frontend tool
    ERROR             = "error"          # fatal; WS will be closed


@dataclass
class LiveEvent:
    """Typed event pushed onto the queue by the receive loop."""

    type: LiveEventType
    data: Optional[str] = None          # base64 audio or transcript text
    mime_type: Optional[str] = None     # e.g. "audio/pcm;rate=24000"
    handle: Optional[str] = None        # reserved (internal use)
    time_left_ms: Optional[int] = None  # for GO_AWAY
    message: Optional[str] = None       # for ERROR / RECONNECTING
    # ── Tool call fields (TOOL_CALL events only) ──────────────────────────────
    call_id: Optional[str] = None       # opaque ID — must be echoed in tool_result
    name: Optional[str] = None          # window.__duconAPI.<name>
    args: Optional[dict] = None         # keyword arguments for the function

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ── Session ────────────────────────────────────────────────────────────────────
class GeminiLiveSession:
    """
    Manages a self-healing real-time conversation with the Gemini Live API.

    Usage::

        session = GeminiLiveSession(user_id=42, resumption_handle="abc")
        q: asyncio.Queue[LiveEvent] = asyncio.Queue()
        await session.start(q)   # connects + signals CONNECTED event

        await session.send_audio(b64_pcm)
        await session.send_text("hello")

        # drain q in a separate task and forward to WebSocket
        event = await q.get()

        await session.close()   # clean shutdown

    Parameters
    ----------
    user_id:
        Optional authenticated user ID (for logging / future personalisation).
    system_instruction:
        Custom system prompt.  Defaults to the Ducon assistant persona.
    resumption_handle:
        Opaque token from a previous session's SESSION_HANDLE event.
        The session runner also updates this value automatically so that
        every reconnect benefits from the most recent Gemini context.
    model:
        Live model ID.  Defaults to the LIVE_MODEL env variable.
    """

    def __init__(
        self,
        *,
        user_id: Optional[int] = None,
        guest_session_id: Optional[str] = None,
        system_instruction: Optional[str] = None,
        resumption_handle: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.user_id = user_id
        self.guest_session_id = guest_session_id
        self.system_instruction = system_instruction or get_live_system_instruction()
        self.resumption_handle = resumption_handle
        self.model = model or cfg("LIVE_MODEL", LIVE_MODEL)

        self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        # The live SDK session object, only valid while runner is connected.
        self._session: Optional[object] = None

        # Signals the session is up and ready to receive sends.
        self._connected: asyncio.Event = asyncio.Event()

        self._event_queue: Optional[asyncio.Queue] = None
        self._runner_task: Optional[asyncio.Task] = None
        self._closed: bool = False

        # ── Pending tool calls ────────────────────────────────────────────────
        # Maps call_id → asyncio.Task (the timeout task).
        # Cleared / cancelled when submit_tool_result is called or on reconnect.
        self._pending_tool_calls: dict[str, asyncio.Task] = {}

        # ── Conversation history ───────────────────────────────────────────────
        # Completed turns in Gemini Content dict format:
        #   [{"role": "user", "parts": [{"text": "..."}]},
        #    {"role": "model", "parts": [{"text": "..."}]}, ...]
        # Re-seeded into every new Gemini session so context is never lost
        # when the underlying WSS reconnects.
        self._conversation_history: list[dict] = []

        # Accumulate transcript chunks for the turn currently in progress.
        # Committed to _conversation_history on each turn_complete.
        self._pending_input: list[str] = []   # user's words for this turn
        self._pending_output: list[str] = []  # model's words for this turn

        # How many tool calls the model actually made in the current turn.
        # Reset on every turn_complete.  Used to detect "hallucinated" tool
        # calls where the model described calling a tool but never did.
        self._tools_called_this_turn: int = 0

        # Number of consecutive [execute_now] retries sent.  Capped at
        # _MAX_EXECUTE_NOW_RETRIES to prevent infinite re-prompt loops.
        self._execute_now_retries: int = 0

        # Raw PCM bytes of audio queued to the frontend in the current turn.
        # Used to estimate playback duration so [execute_now] is not sent
        # while the audio is still playing (PCM 24 kHz 16-bit = 48 000 B/s).
        self._pending_audio_bytes: int = 0
        # Saved at turn_complete so _maybe_retry_tool_call can read it after
        # _pending_audio_bytes has been reset for the next turn.
        self._last_turn_audio_bytes: int = 0

    def seed_history(self, turns: list[dict]) -> None:
        """
        Pre-load chat transcript turns before the first Live connect.

        Used to sync text-chat context into a new voice session (chat → voice).
        Each turn must be a Gemini Content dict: {role, parts: [{text}]}. 
        """
        if not turns:
            return
        self._conversation_history = list(turns)[-_MAX_HISTORY_TURNS:]

    # ── Config ────────────────────────────────────────────────────────────────

    def _build_config(self) -> dict:
        """
        Build the LiveConnectConfig dict.

        Using a plain dict decouples us from SDK class names that change
        between minor releases.
        """
        _voice = cfg("LIVE_VOICE", LIVE_VOICE)
        _trigger = int(cfg("LIVE_COMPRESSION_TRIGGER", _COMPRESSION_TRIGGER_TOKENS))
        _target = int(cfg("LIVE_COMPRESSION_TARGET", _COMPRESSION_TARGET_TOKENS))
        _thinking = cfg("LIVE_THINKING_LEVEL", LIVE_THINKING_LEVEL)
        config: dict = {
            # Only AUDIO output is supported on native-audio live models.
            # Text is accessed via output_audio_transcription.
            "response_modalities": ["AUDIO"],

            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": _voice}
                }
            },

            # Transcribe both sides so the UI can show captions.
            "input_audio_transcription": {},
            "output_audio_transcription": {},

            # Automatic VAD — server detects speech boundaries and barge-in.
            # silence_duration_ms controls how long of silence ends a turn.
            # prefix_padding_ms includes audio before speech starts (avoids
            # clipping the first syllable).
            "realtime_input_config": {
                "automatic_activity_detection": {
                    "disabled": False,
                    "prefix_padding_ms": _vad_prefix_ms(),
                    "silence_duration_ms": _vad_silence_ms(),
                }
            },

            # Sliding-window context compression — unlimited session duration.
            "context_window_compression": {
                "trigger_tokens": _trigger,
                "sliding_window": {"target_tokens": _target},
            },

            # Session resumption — receive periodic handles that let us
            # reconnect after the ~10-minute WSS connection lifetime.
            "session_resumption": {
                "handle": self.resumption_handle,
            },

            # Required for gemini-3.1-flash-live-preview to accept
            # send_client_content for conversation history seeding on reconnect.
            "history_config": {
                "initial_history_in_client_content": True,
            },

            # Frontend tools — executed via window.__duconAPI.* on the client.
            # Gemini will emit tool_call messages; we bridge them to the client
            # and relay the tool_result back with submit_tool_result().
            "tools": _TOOLS,
        }

        if self.system_instruction:
            config["system_instruction"] = self.system_instruction

        if _thinking and _thinking.lower() not in ("none", ""):
            config["thinking_config"] = {"thinking_level": _thinking}

        return config

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self, event_queue: asyncio.Queue) -> None:
        """
        Launch the session runner and wait until the first connection is
        established (or raise on timeout).
        """
        if self._closed:
            raise RuntimeError("Session has already been closed.")

        self._event_queue = event_queue
        self._runner_task = asyncio.create_task(
            self._session_runner(),
            name=f"gemini-live-runner-user{self.user_id}",
        )

        try:
            await asyncio.wait_for(self._connected.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            await self.close()
            raise RuntimeError("Timed out waiting for Gemini Live connection.")

        logger.info(
            "Gemini Live session ready — user=%s model=%s voice=%s",
            self.user_id, self.model, cfg("LIVE_VOICE", LIVE_VOICE),
        )

    async def close(self) -> None:
        """Cancel the runner task.  The `async with` block exits cleanly."""
        if self._closed:
            return
        self._closed = True
        self._connected.clear()

        # Cancel any outstanding tool-call timeout tasks.
        for task in list(self._pending_tool_calls.values()):
            task.cancel()
        self._pending_tool_calls.clear()

        if self._runner_task and not self._runner_task.done():
            self._runner_task.cancel()
            try:
                await self._runner_task
            except (asyncio.CancelledError, Exception):
                pass

        logger.info("Gemini Live session closed — user=%s", self.user_id)

    # ── Send helpers ───────────────────────────────────────────────────────────

    async def _wait_for_session(self) -> Optional[object]:
        """
        Return the active Gemini session, waiting up to SEND_WAIT_TIMEOUT
        seconds if the runner is mid-reconnect.

        We check BOTH self._session (not None) AND self._connected (is set)
        to avoid returning a stale session object that still exists in memory
        but whose underlying WebSocket is already closed.

        Returns None only when the session is permanently closed or the
        timeout expires.
        """
        if self._closed:
            return None
        # Both guards must pass: object exists AND connection flag is set.
        if self._session and self._connected.is_set():
            return self._session
        # Session is reconnecting — wait for it to come back.
        _swt = _send_wait_timeout()
        logger.info(
            "send: session reconnecting, waiting up to %.0fs — user=%s",
            _swt, self.user_id,
        )
        try:
            await asyncio.wait_for(
                asyncio.shield(self._connected.wait()), timeout=_swt
            )
        except asyncio.TimeoutError:
            logger.warning(
                "send: gave up waiting for session reconnect — user=%s", self.user_id
            )
            return None
        # Re-check after wait; close() may have raced us.
        if self._session and self._connected.is_set():
            return self._session
        return None

    async def send_audio(
        self, b64_pcm: str, mime_type: str = "audio/pcm;rate=16000"
    ) -> None:
        """Forward a base64-encoded PCM audio chunk (16-bit 16 kHz mono)."""
        session = await self._wait_for_session()
        if not session:
            return
        try:
            audio_bytes = base64.b64decode(b64_pcm)
            await session.send_realtime_input(  # type: ignore[attr-defined]
                audio=types.Blob(data=audio_bytes, mime_type=mime_type)
            )
        except Exception as exc:
            logger.warning("send_audio failed: %s — user=%s", exc, self.user_id)

    async def send_text(self, text: str) -> None:
        """
        Send a text message mid-conversation.

        For gemini-3.1-flash-live-preview, mid-session text must go via
        send_realtime_input (not send_client_content).
        """
        session = await self._wait_for_session()
        if not session:
            logger.warning("send_text dropped (no session): %r — user=%s", text, self.user_id)
            return
        logger.info("send_text → Gemini: %r — user=%s", text[:80], self.user_id)
        _dbg(f"[LIVE ▶ SEND] text → \"{text[:120]}\"  (user={self.user_id})")
        # Text input is not transcribed by the ASR pipeline, so we track it
        # manually so it ends up in the conversation history.
        self._pending_input.append(text)
        try:
            await session.send_realtime_input(text=text)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("send_text failed: %s — user=%s", exc, self.user_id)

    async def send_audio_stream_end(self) -> None:
        """Signal mic pause so Gemini can flush its audio buffer."""
        session = await self._wait_for_session()
        if not session:
            return
        logger.debug("send_audio_stream_end — user=%s", self.user_id)
        try:
            await session.send_realtime_input(audio_stream_end=True)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("send_audio_stream_end failed: %s — user=%s", exc, self.user_id)

    async def submit_tool_result(
        self,
        call_id: str,
        name: str,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Forward the result of a frontend tool execution back to Gemini.

        Called by voice.py when it receives a ``tool_result`` message from
        the client WebSocket.  Gemini pauses its turn while waiting for this;
        once received it will continue (or speak) with the tool output.

        Parameters
        ----------
        call_id:
            The opaque ID that arrived in the TOOL_CALL event — must match.
        name:
            The function name, also from the TOOL_CALL event.
        result:
            Arbitrary JSON-serialisable dict returned by the frontend tool.
        error:
            Human-readable error string if the tool failed; mutually exclusive
            with *result*.
        """
        # Cancel the timeout task for this call so it doesn't also auto-respond.
        timeout_key = call_id or name
        timeout_task = self._pending_tool_calls.pop(timeout_key, None)
        if timeout_task and not timeout_task.done():
            timeout_task.cancel()
            try:
                await timeout_task
            except asyncio.CancelledError:
                pass

        session = await self._wait_for_session()
        if not session:
            logger.warning(
                "submit_tool_result: no session — dropping %s — user=%s",
                name, self.user_id,
            )
            return

        # Sanitize and normalise: Gemini's FunctionResponse.response must be a
        # dict.  Some tools (e.g. AISearch) return a list — wrap it so the SDK
        # never receives a bare list.
        compact_result = _compact_tool_result(name, result or {}) if not error else {}
        raw_sanitised = _sanitize_tool_result(compact_result) if not error else {}
        if isinstance(raw_sanitised, list):
            safe_result: dict = {"items": raw_sanitised}
        elif isinstance(raw_sanitised, dict):
            safe_result = raw_sanitised
        else:
            safe_result = {"value": raw_sanitised}

        response_payload: dict = {"error": error} if error else {"result": safe_result}
        result_summary = (
            f"keys={list(safe_result.keys())}"
            if isinstance(safe_result, dict)
            else repr(safe_result)[:60]
        )
        logger.info(
            "→ Gemini tool_response: %s call_id=%s error=%s %s — user=%s",
            name, call_id, bool(error), result_summary, self.user_id,
        )
        id_val = safe_result.get("id") if isinstance(safe_result, dict) else None
        _dbg(f"[LIVE ▶ TOOL RESULT] {name}  error={bool(error)}  id={id_val}  {result_summary}  (user={self.user_id})")
        try:
            await session.send_tool_response(  # type: ignore[attr-defined]
                function_responses=[
                    types.FunctionResponse(
                        id=call_id,
                        name=name,
                        response=response_payload,
                    )
                ]
            )
        except Exception as exc:
            logger.warning(
                "submit_tool_result failed: %s — user=%s", exc, self.user_id
            )

    def cancel_tool_timeout(self, call_id: str) -> None:
        """
        Cancel the pending auto-timeout for a tool call.

        Called by voice.py when it receives a ``tool_pause`` message from the
        client — meaning a human-input tool (e.g. UploadImage) is waiting for
        the user to interact and the 30-second safety timer must not fire.
        The normal ``tool_result`` message will still arrive once the user is done.
        """
        timeout_key = call_id or ""
        task = self._pending_tool_calls.pop(timeout_key, None)
        if task and not task.done():
            task.cancel()
            logger.info(
                "Tool timeout cancelled (waiting for user input) call_id=%s — user=%s",
                call_id, self.user_id,
            )

    async def _tool_call_timeout(self, call_id: str, name: str) -> None:
        """
        Auto-respond to a tool call with an error if the frontend doesn't
        reply within _TOOL_CALL_TIMEOUT_SECS.

        Gemini 3.1 Flash Live uses synchronous-only function calling: the
        model halts all output and waits for a FunctionResponse.  Without
        this timeout, a missing or slow frontend handler would freeze the
        entire voice session indefinitely.
        """
        _timeout = _tool_call_timeout()
        await asyncio.sleep(_timeout)

        timeout_key = call_id or name
        if timeout_key not in self._pending_tool_calls:
            return  # submit_tool_result already handled this call
        self._pending_tool_calls.pop(timeout_key, None)

        logger.warning(
            "Tool call timed out after %.0fs: %s call_id=%s — "
            "auto-responding with error so Gemini can continue — user=%s",
            _timeout, name, call_id, self.user_id,
        )
        session = self._session
        if session is None:
            return
        try:
            await session.send_tool_response(  # type: ignore[attr-defined]
                function_responses=[
                    types.FunctionResponse(
                        id=call_id,
                        name=name,
                        response={"error": "Tool call timed out — no response from client"},
                    )
                ]
            )
        except Exception as exc:
            logger.warning(
                "Tool-timeout auto-response failed: %s — user=%s", exc, self.user_id
            )

    async def _maybe_retry_tool_call(self, model_output: str, audio_bytes: int) -> None:
        """
        Classify whether the model announced a tool action but missed calling
        it, then send [execute_now] if confirmed.

        Runs as a background task after every TURN_COMPLETE where no tool was
        called.  Uses a lightweight LLM call so it works in any language.

        `audio_bytes` is the raw PCM byte count for the completed turn.  It is
        passed to _send_execute_now to compute a playback-aware delay.
        """
        if self._closed:
            return
        if self._execute_now_retries >= _MAX_EXECUTE_NOW_RETRIES:
            _dbg(f"[LIVE ⚠ RETRY LIMIT] max execute_now retries reached — skipping classifier  (user={self.user_id})")
            return

        intent_detected = await _classify_missed_tool_call(model_output, self._client)
        if not intent_detected:
            self._execute_now_retries = 0  # clean conversational turn
            return

        self._execute_now_retries += 1
        logger.info(
            "Missed tool call detected (retry %d/%d) — sending [execute_now] — user=%s",
            self._execute_now_retries, _MAX_EXECUTE_NOW_RETRIES, self.user_id,
        )
        _dbg(f"[LIVE ⚡ RETRY {self._execute_now_retries}/{_MAX_EXECUTE_NOW_RETRIES}] intent confirmed — sending [execute_now]  (user={self.user_id})")
        await self._send_execute_now(audio_bytes)

    async def _send_execute_now(self, audio_bytes: int = 0) -> None:
        """
        Send a silent [execute_now] trigger so the model retries calling the
        tool it announced in the previous turn.

        Delays long enough for the frontend to finish playing the turn audio
        before a new Gemini turn begins.

        Delay formula (PCM 24 kHz, 16-bit → 48 000 bytes/sec):
            delay = audio_duration_secs + 1.5 s buffer, min 0.8 s
        The 1.5 s buffer accounts for network round-trip, browser buffering,
        and the time the classifier itself already consumed.
        """
        _PCM_BYTES_PER_SEC = 48_000  # 24 kHz × 2 bytes (16-bit mono)
        audio_secs = audio_bytes / _PCM_BYTES_PER_SEC
        delay = max(0.8, audio_secs + 1.5)
        _dbg(f"[LIVE ⏱ DELAY] execute_now waiting {delay:.1f}s (audio={audio_secs:.1f}s)  (user={self.user_id})")
        await asyncio.sleep(delay)
        if self._closed:
            return
        session = await self._wait_for_session()
        if not session:
            return
        try:
            await session.send_realtime_input(text="[execute_now]")  # type: ignore[attr-defined]
            logger.info("execute_now trigger sent — user=%s", self.user_id)
        except Exception as exc:
            logger.warning("execute_now trigger failed: %s — user=%s", exc, self.user_id)

    # ── Session runner (the heart of the class) ────────────────────────────────

    async def _session_runner(self) -> None:
        """
        Establish and maintain the Gemini Live connection.

        This coroutine owns the single `async with client.aio.live.connect()`
        block — the only safe way to use the SDK.

        Reconnect behaviour
        -------------------
        The loop runs until close() cancels the task.  Whether the session
        ends with an exception (keepalive drop, network error) or a clean code
        1000 close (GoAway, 10-minute limit, stale resumption handle), we
        always reconnect.

        Stale resumption handles
        -------------------------
        If the session closes normally and Gemini did NOT issue a fresh handle
        during that session, the handle we connected with is considered consumed
        / stale.  We clear it so the next reconnect starts a completely fresh
        session rather than endlessly repeating a completed conversation.
        """
        backoff = _RECONNECT_INITIAL_DELAY

        while not self._closed:
            # ── Clear stale state from the previous iteration ─────────────────
            self._session = None
            self._connected.clear()

            # Remember the handle we're ENTERING this connection with.
            # After the session, if Gemini issued a newer handle, it will be
            # in self.resumption_handle; otherwise the old one is stale.
            handle_at_entry = self.resumption_handle
            received_new_handle = False

            try:
                async with self._client.aio.live.connect(
                    model=self.model, config=self._build_config()
                ) as session:
                    backoff = _RECONNECT_INITIAL_DELAY  # reset on success

                    # ── Seed conversation history BEFORE signalling ready ──────
                    # Re-injecting the transcript means Gemini picks up exactly
                    # where it left off even if the WSS had to be restarted.
                    # We do this BEFORE setting _connected so no audio can race
                    # in while we're still loading the context.
                    if self._conversation_history:
                        try:
                            await session.send_client_content(  # type: ignore[attr-defined]
                                turns=self._conversation_history,
                                turn_complete=False,
                            )
                            logger.info(
                                "History seeded: %d turns — user=%s",
                                len(self._conversation_history), self.user_id,
                            )
                        except Exception as seed_exc:
                            logger.warning(
                                "History seeding failed (continuing fresh): %s — user=%s",
                                seed_exc, self.user_id,
                            )

                    # Now open the gate — sends and the frontend CONNECTED
                    # event fire only after history is in place.
                    self._session = session
                    self._connected.set()

                    if self._event_queue:
                        await self._event_queue.put(
                            LiveEvent(type=LiveEventType.CONNECTED)
                        )

                    logger.info(
                        "Gemini Live connected — user=%s model=%s handle=%s history_turns=%d",
                        self.user_id, self.model, bool(handle_at_entry),
                        len(self._conversation_history),
                    )
                    _dbg(f"[LIVE ✓ CONN] Gemini Live connected  model={self.model}  history={len(self._conversation_history)} turns  (user={self.user_id})")

                    # ── Auto-greeting on fresh sessions ───────────────────────
                    # On the very first connection (no conversation history yet)
                    # we send a hidden trigger via send_realtime_input — the
                    # standard input path for native-audio Live models — so the
                    # model speaks its greeting immediately without waiting for
                    # the user to say something first.
                    # On reconnects _conversation_history is already populated,
                    # so use a resume trigger instead of the fresh-session greeting.
                    if not self._conversation_history:
                        trigger = "[session_start]"
                    else:
                        trigger = "[session_resume]"
                    try:
                        await session.send_realtime_input(  # type: ignore[attr-defined]
                            text=trigger,
                        )
                        logger.info(
                            "Session trigger sent (%s) — user=%s history_turns=%d",
                            trigger, self.user_id, len(self._conversation_history),
                        )
                        _dbg(f"[LIVE ▶ SEND] trigger → {trigger}  history={len(self._conversation_history)}  (user={self.user_id})")
                    except Exception as greet_exc:
                        logger.warning(
                            "Session trigger failed: %s — user=%s",
                            greet_exc, self.user_id,
                        )
                        _dbg(f"[LIVE ✗ SEND] trigger FAILED: {greet_exc}  (user={self.user_id})")

                    # ── Multi-turn receive loop ───────────────────────────────
                    # Per the docs, session.receive() processes ONE complete
                    # model turn and breaks after turn_complete.  We wrap it in
                    # a while loop to keep the same underlying WebSocket alive
                    # across consecutive turns (no close/reconnect overhead and
                    # no loss of Gemini's own in-session context).
                    #
                    # If the WebSocket drops mid-turn (keepalive timeout, GoAway
                    # expiry, network error), receive() raises an exception that
                    # propagates out of this while loop and is caught below →
                    # the outer loop reconnects transparently.
                    while not self._closed:
                        async for message in session.receive():  # type: ignore[attr-defined]
                            logger.debug(
                                "← Gemini raw: %s — user=%s",
                                _summarise_message(message), self.user_id,
                            )
                            if self.resumption_handle != handle_at_entry:
                                received_new_handle = True
                            await self._dispatch(message)

                        # turn_complete received — session WebSocket still open.
                        if not self._closed:
                            logger.debug(
                                "Turn complete — session alive, ready for next turn — user=%s",
                                self.user_id,
                            )

                    logger.info(
                        "Gemini Live multi-turn loop exited — user=%s", self.user_id
                    )

            except asyncio.CancelledError:
                # close() was called — exit the loop cleanly.
                self._session = None
                self._connected.clear()
                return

            except Exception as exc:
                logger.warning(
                    "Gemini Live connection dropped — user=%s error=%s "
                    "reconnecting in %.1fs",
                    self.user_id, exc, backoff,
                )
                _dbg(f"[LIVE ✗ CONN] connection dropped: {exc}  reconnecting in {backoff:.1f}s  (user={self.user_id})")
                if self._event_queue:
                    await self._event_queue.put(
                        LiveEvent(
                            type=LiveEventType.RECONNECTING,
                            message=f"Reconnecting in {backoff:.0f}s…",
                        )
                    )

            # ── Post-connection cleanup ───────────────────────────────────────
            self._session = None
            self._connected.clear()

            # Discard any partial turn (the session closed mid-response).
            # Only fully committed turns (via turn_complete) stay in history.
            self._pending_input.clear()
            self._pending_output.clear()

            # Cancel outstanding tool-call timeouts — they're meaningless once
            # the connection is gone (we'll reconnect with fresh history).
            for task in list(self._pending_tool_calls.values()):
                task.cancel()
            self._pending_tool_calls.clear()

            if self._closed:
                return

            # If we didn't receive a fresh resumption handle during this
            # session, the one we came in with is consumed / stale.  Clear it
            # so the next attempt starts a fresh conversation.
            if not received_new_handle and handle_at_entry:
                logger.info(
                    "Clearing stale resumption handle — user=%s", self.user_id
                )
                self.resumption_handle = None

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_DELAY)

    # ── Message dispatcher ─────────────────────────────────────────────────────

    async def _dispatch(self, message: object) -> None:  # noqa: C901
        """Parse a LiveServerMessage and push typed events onto the queue."""
        q = self._event_queue
        if q is None:
            return

        # ── Session resumption handle ──────────────────────────────────────────
        # Stored internally so the _session_runner can reconnect transparently
        # when the ~10-minute Gemini WSS limit is hit.  NOT forwarded to the
        # frontend — handles are an implementation detail of the backend.
        # Exposing them would let the frontend reuse handles across separate
        # conversations, carrying over unwanted history.
        update = getattr(message, "session_resumption_update", None)
        if update and getattr(update, "resumable", False):
            handle = getattr(update, "new_handle", None)
            if handle:
                self.resumption_handle = handle
                logger.debug(
                    "Resumption handle updated (internal) — user=%s", self.user_id
                )

        # ── GoAway ─────────────────────────────────────────────────────────────
        go_away = getattr(message, "go_away", None)
        if go_away is not None:
            time_left = getattr(go_away, "time_left", None)
            ms_left = int(time_left.total_seconds() * 1000) if time_left else 0
            await q.put(LiveEvent(type=LiveEventType.GO_AWAY, time_left_ms=ms_left))
            logger.info(
                "GoAway — user=%s time_left_ms=%d handle=%s",
                self.user_id, ms_left, self.resumption_handle,
            )
            _dbg(f"[LIVE ⚠ AWAY ] session closing in {ms_left/1000:.0f}s — will reconnect with resumption handle  (user={self.user_id})")

        # ── Usage metadata ─────────────────────────────────────────────────────
        # Gemini periodically sends token-usage snapshots.  We print them so
        # you can monitor context growth and see when compression fires.
        usage = getattr(message, "usage_metadata", None)
        if usage is not None:
            total   = getattr(usage, "total_token_count", None)
            prompt  = getattr(usage, "prompt_token_count", None)
            cached  = getattr(usage, "cached_content_token_count", None)
            cand    = getattr(usage, "candidates_token_count", None)
            _trig = int(cfg("LIVE_COMPRESSION_TRIGGER", _COMPRESSION_TRIGGER_TOKENS))
            _tgt = int(cfg("LIVE_COMPRESSION_TARGET", _COMPRESSION_TARGET_TOKENS))
            if total is not None:
                pct = int(total / _trig * 100) if _trig else 0
                bar_len = 20
                filled  = int(bar_len * total / _trig) if _trig else 0
                bar     = "█" * min(filled, bar_len) + "░" * (bar_len - min(filled, bar_len))
                _dbg(
                    f"[LIVE ◀ USAGE] [{bar}] {total:,} / {_trig:,} tokens ({pct}%)"
                    f"  prompt={prompt}  output={cand}  cached={cached}"
                    f"  (user={self.user_id})"
                )
                # Non-blocking usage/cost recording for the live voice agent.
                try:
                    from app.admin.usage_recorder import record
                    record(
                        agent="live", model=self.model, provider="gemini",
                        user_id=self.user_id, guest_session_id=self.guest_session_id,
                        input_tokens=int(prompt or 0), output_tokens=int(cand or 0),
                    )
                except Exception:
                    pass
                if total >= _trig:
                    _dbg(f"[LIVE ⚡ COMP ] compression triggered — target {_tgt:,} tokens  (user={self.user_id})")
                elif total >= int(_trig * 0.8):
                    _dbg(f"[LIVE ⚠ USAGE] approaching compression threshold ({pct}% full)  (user={self.user_id})")

        # ── Server content BEFORE tool calls ───────────────────────────────────
        # Process server_content (audio chunks, transcripts, interrupted,
        # turn_complete) BEFORE tool_call so that any narration audio the model
        # emitted in the same SDK message is queued and delivered to the client
        # before the TOOL_CALL event arrives.  This preserves the correct
        # playback order: the client plays the narration, then executes the tool.
        sc = getattr(message, "server_content", None)
        if sc is not None:
            # Barge-in: model was interrupted by user speech
            if getattr(sc, "interrupted", False):
                logger.info("← Gemini: INTERRUPTED — user=%s", self.user_id)
                _dbg(f"[LIVE ◀ RECV] INTERRUPTED  (user={self.user_id})")
                self._pending_audio_bytes = 0  # audio discarded; reset counter
                await q.put(LiveEvent(type=LiveEventType.INTERRUPTED))

            # Model turn: audio chunks and/or text parts
            model_turn = getattr(sc, "model_turn", None)
            if model_turn:
                for part in (getattr(model_turn, "parts", None) or []):
                    inline = getattr(part, "inline_data", None)
                    if inline and inline.data:
                        self._pending_audio_bytes += len(inline.data)
                        b64 = base64.b64encode(inline.data).decode()
                        await q.put(LiveEvent(
                            type=LiveEventType.AUDIO_CHUNK,
                            data=b64,
                            mime_type="audio/pcm;rate=24000",
                        ))
                    text = getattr(part, "text", None)
                    if text:
                        await q.put(LiveEvent(
                            type=LiveEventType.OUTPUT_TRANSCRIPT, data=text
                        ))

            # Input transcription (ASR of what the user said)
            # Accumulated across partial events; committed to history on turn_complete.
            in_tx = getattr(sc, "input_transcription", None)
            if in_tx:
                tx_text = getattr(in_tx, "text", None)
                if tx_text:
                    self._pending_input.append(tx_text)
                    logger.info(
                        "← Gemini input_transcript: %r — user=%s", tx_text[:80], self.user_id
                    )
                    _dbg(f"[LIVE ◀ USER ] \"{tx_text}\"")
                    await q.put(LiveEvent(
                        type=LiveEventType.INPUT_TRANSCRIPT, data=tx_text
                    ))

            # Output transcription (what the model is saying, as text)
            # Accumulated for history; forwarded to frontend for captions.
            out_tx = getattr(sc, "output_transcription", None)
            if out_tx:
                tx_text = getattr(out_tx, "text", None)
                if tx_text:
                    self._pending_output.append(tx_text)
                    _dbg(f"[LIVE ◀ MODEL] \"{tx_text}\"")
                    await q.put(LiveEvent(
                        type=LiveEventType.OUTPUT_TRANSCRIPT, data=tx_text
                    ))

            # Turn complete — commit the accumulated transcript to history.
            if getattr(sc, "turn_complete", False):
                # Grab the model's output text BEFORE _commit_turn clears it.
                model_output = "".join(self._pending_output).lower()
                # Save audio bytes then reset for the next turn.
                self._last_turn_audio_bytes = self._pending_audio_bytes
                self._pending_audio_bytes = 0

                self._commit_turn()
                logger.info(
                    "← Gemini: TURN_COMPLETE — user=%s history_turns=%d",
                    self.user_id, len(self._conversation_history),
                )
                _dbg(f"[LIVE ◀ RECV] TURN_COMPLETE  history={len(self._conversation_history)} turns  tools_called={self._tools_called_this_turn}  (user={self.user_id})")

                # ── Missed-tool-call recovery (known Gemini Live bug) ──────────
                # The model sometimes announces it will call a tool in audio but
                # the actual tool_call message is never emitted and the turn
                # completes.  This is a confirmed intermittent model bug (GitHub
                # googleapis/python-genai #813).
                #
                # Detection: a lightweight async LLM classifier (gemini-2.0-flash-lite)
                # reads the model's transcript and answers "did this turn announce
                # an action without executing it?".  Works in any language.
                if self._tools_called_this_turn == 0:
                    asyncio.create_task(
                        self._maybe_retry_tool_call(model_output, self._last_turn_audio_bytes),
                        name=f"intent-check-{self.user_id}",
                    )
                else:
                    # Tool was actually called — reset retry counter.
                    self._execute_now_retries = 0

                self._tools_called_this_turn = 0  # reset for next turn

                await q.put(LiveEvent(type=LiveEventType.TURN_COMPLETE))

        # ── Tool calls — always after server_content ───────────────────────────
        # Queued last so that any narration audio already in the queue reaches
        # the client and starts playing before the tool_call event triggers
        # frontend execution.
        #
        # IMPORTANT for the frontend:
        #   On receipt of a tool_call event, drain and finish playing the entire
        #   audio buffer before calling window.__duconAPI[name](args).  The
        #   narration audio describing the action is already in the pipeline and
        #   must play to completion so the user hears it.
        #
        # Timeout policy:
        #   Tools in _NO_TIMEOUT_TOOLS (UploadImage, generate_multi_image) never
        #   get an auto-timeout.  UploadImage is user-paced; generate_multi_image
        #   can take 60–120 s.  All other tools use _TOOL_CALL_TIMEOUT_SECS.
        tool_call = getattr(message, "tool_call", None)
        if tool_call:
            function_calls = getattr(tool_call, "function_calls", None) or []
            for fc in function_calls:
                call_id = getattr(fc, "id", None) or ""
                name = getattr(fc, "name", "") or ""
                raw_args = getattr(fc, "args", None)
                try:
                    # Round-trip through JSON to guarantee a plain, fully
                    # serialisable dict — some SDK versions expose proto-backed
                    # Struct objects that look like dicts but fail json.dumps.
                    args: dict = json.loads(json.dumps(raw_args)) if raw_args else {}
                except (TypeError, ValueError):
                    args = {}

                self._tools_called_this_turn += 1

                if name in _NO_TIMEOUT_TOOLS:
                    logger.info(
                        "← Gemini tool_call: %s(%s) call_id=%s [no timeout] — user=%s",
                        name, list(args.keys()), call_id, self.user_id,
                    )
                    _dbg(f"[LIVE ◀ TOOL ] {name}({args})  [no timeout]  (user={self.user_id})")
                else:
                    _tct = _tool_call_timeout()
                    logger.info(
                        "← Gemini tool_call: %s(%s) call_id=%s [timeout=%.0fs] — user=%s",
                        name, list(args.keys()), call_id,
                        _tct, self.user_id,
                    )
                    _dbg(f"[LIVE ◀ TOOL ] {name}({args})  [timeout={_tct:.0f}s]  (user={self.user_id})")
                    # Spawn a timeout so that if the frontend never replies,
                    # Gemini gets an auto error-response instead of freezing.
                    timeout_key = call_id or name
                    self._pending_tool_calls[timeout_key] = asyncio.create_task(
                        self._tool_call_timeout(call_id, name),
                        name=f"tool-timeout-{timeout_key[:12]}",
                    )

                await q.put(LiveEvent(
                    type=LiveEventType.TOOL_CALL,
                    call_id=call_id,
                    name=name,
                    args=args,
                ))



    def _commit_turn(self) -> None:
        """
        Commit the pending input/output transcript to conversation history.

        Called on every turn_complete.  Both sides of the turn are stored as
        separate Content dicts so they can be re-fed to Gemini with
        send_client_content on reconnect.
        """
        user_text = "".join(self._pending_input).strip()
        model_text = "".join(self._pending_output).strip()

        if user_text:
            self._conversation_history.append({
                "role": "user",
                "parts": [{"text": user_text}],
            })
        if model_text:
            self._conversation_history.append({
                "role": "model",
                "parts": [{"text": model_text}],
            })

        # Mirror completed voice turns into the shared chat transcript so a
        # later voice session can seed chat context (chat ↔ voice sync).
        if user_text or model_text:
            if self.user_id is not None:
                from app import chat_session
                chat_session.append_turn(self.user_id, user_text, model_text)
            elif self.guest_session_id:
                from app import chat_session
                chat_session.append_guest_turn(
                    self.guest_session_id, user_text, model_text,
                )

        # Trim oldest turns to stay within _MAX_HISTORY_TURNS.
        if len(self._conversation_history) > _MAX_HISTORY_TURNS:
            self._conversation_history = (
                self._conversation_history[-_MAX_HISTORY_TURNS:]
            )

        self._pending_input.clear()
        self._pending_output.clear()


# ── Helpers ────────────────────────────────────────────────────────────────────

# Keys that are known to carry binary / blob / URL data that Gemini Live
# cannot accept inside a FunctionResponse (causes 1007 invalid argument).
_BINARY_KEYS = frozenset({
    "blob", "blobUrl", "blobData", "imageData", "image_data",
    "imgData", "img_data", "fileData", "file_data", "bytes",
    "buffer", "arrayBuffer",
})
# Maximum length for any single string value passed to Gemini.
# Data-URIs and base64 payloads are typically much larger than this.
_MAX_STR_LEN = 512


def _compact_designer_job_result(result: object) -> dict:
    """
    Return only the essential fields from a start_designer_job result so the
    voice model can describe the outcome without hitting the 12 k char limit.
    The full evaluation/references metadata is stripped; the model can call
    get_image(generation_ref) to open the result if needed.
    """
    if not isinstance(result, dict):
        return {"raw": str(result)[:500]}

    best: dict = result.get("best_generation") or {}
    gen_id = best.get("id")
    refs   = result.get("references") or []

    return {
        "job_id":          result.get("job_id"),
        "status":          result.get("status"),
        "summary":         (result.get("summary") or "")[:1200],
        "best_generation": {
            "id":              gen_id,
            "generation_name": best.get("generation_name"),
        } if gen_id else None,
        # Canonical ref the model should pass to get_image() to open the result
        "generation_ref":  f"gen:{gen_id}" if gen_id else None,
        "references_used": [
            {
                "id": r.get("id"),
                "name": r.get("name") or r.get("label"),
                "catalog_ref": r.get("id"),
            }
            for r in refs
            if isinstance(r, dict) and r.get("id")
        ],
        "attempts": len(result.get("attempts") or []),
        "note": (
            "Call get_image(generation_ref) to open the generated design. "
            "When the user asks which Ducon catalog images were used, list references_used "
            "above — do not search the catalog again."
        ),
    }


def _compact_send_to_chat_result(result: object) -> dict:
    """Compact delegation result for the voice model."""
    if not isinstance(result, dict):
        return {"raw": str(result)[:500]}

    designer = result.get("designer_job")
    if isinstance(designer, dict):
        compact = _compact_designer_job_result(designer)
        compact["delegated_via"] = "send_to_chat"
        if result.get("voice_reply"):
            compact["voice_reply"] = str(result.get("voice_reply"))[:1200]
        if result.get("assistant_summary") and not compact.get("summary"):
            compact["summary"] = str(result.get("assistant_summary"))[:1200]
        compact["instruction"] = (
            "Now speak a short natural response to the user using voice_reply or summary. "
            "Say the design is ready in the chat and mention the generation_ref if useful."
        )
        return compact

    return {
        "status":           result.get("status") or "completed",
        "assistant_summary": (result.get("assistant_summary") or "")[:1200],
        "voice_reply":      (result.get("voice_reply") or result.get("assistant_summary") or "")[:1200],
        "generation_ref":   result.get("generation_ref"),
        "interaction_id":   result.get("interaction_id"),
        "delegated_via":    "send_to_chat",
        "note": (
            "The full designer job timeline is in the chat panel. "
            "Call get_image(generation_ref) to open a generated design."
        ),
    }


def _compact_tool_result(name: str, result: object) -> object:
    if name == "send_to_chat":
        return _compact_send_to_chat_result(result)
    if name == "start_designer_job":
        return _compact_designer_job_result(result)

    if name != "AISearch":
        return _truncate_jsonish(result, _live_tool_result_max_chars())

    items = result if isinstance(result, list) else (result or {}).get("items") if isinstance(result, dict) else []
    if not isinstance(items, list):
        return result

    compacted: list[dict] = []
    for item in items[:_live_aisearch_max_items()]:
        if not isinstance(item, dict):
            compacted.append({"value": str(item)[:500]})
            continue
        compacted.append({
            key: item.get(key)
            for key in ("id", "name", "title", "filename", "class", "theme", "level", "project", "tags", "url", "_type")
            if item.get(key) is not None
        })

    return {
        "items": compacted,
        "total_returned": len(items),
        "note": (
            "Compacted AISearch metadata. These are search records, not visual inspection. "
            "Call get_image with an id/name when actual image understanding is needed."
        ),
    }


def _truncate_jsonish(data: object, max_chars: int) -> object:
    try:
        text = json.dumps(data, ensure_ascii=False)
    except TypeError:
        return data
    if len(text) <= max_chars:
        return data
    return {
        "truncated": True,
        "original_chars": len(text),
        "preview": text[:max_chars],
    }


def _sanitize_tool_result(data: object, _depth: int = 0) -> object:
    """
    Recursively strip binary/image fields from a tool result before sending
    to Gemini Live API.

    Gemini's FunctionResponse only accepts structured text/metadata; sending
    inline image bytes or data-URIs triggers a 1007 WebSocket close with
    "Request contains an invalid argument."

    Rules applied at every level:
    • Keys in _BINARY_KEYS are dropped entirely.
    • String values that start with "data:" (data-URIs) are replaced with
      a short placeholder so Gemini knows the field existed.
    • String values longer than _MAX_STR_LEN are truncated — long strings
      are almost always serialised binary blobs.
    • Recursion stops at depth 8 to guard against pathological inputs.
    """
    if _depth > 8:
        return None

    if isinstance(data, dict):
        cleaned: dict = {}
        for k, v in data.items():
            if k in _BINARY_KEYS:
                continue
            sanitised = _sanitize_tool_result(v, _depth + 1)
            if sanitised is not None:
                cleaned[k] = sanitised
        return cleaned

    if isinstance(data, list):
        return [
            s for item in data
            if (s := _sanitize_tool_result(item, _depth + 1)) is not None
        ]

    if isinstance(data, str):
        if data.startswith("data:"):
            # data-URI — replace with placeholder so Gemini knows the field
            # was present without receiving megabytes of base64.
            mime = data[5:data.index(",")] if "," in data else "binary"
            return f"[{mime} data omitted]"
        if len(data) > _MAX_STR_LEN:
            # Likely base64 or serialised binary — truncate it.
            return data[:_MAX_STR_LEN] + "…[truncated]"
        return data

    # int, float, bool, None — pass through unchanged.
    return data


# Minimum word count for a model turn to be worth classifying.
# Very short turns ("Sure!", "Got it.", "Okay.") are never tool-intent turns.
_MIN_WORDS_FOR_INTENT_CHECK = 6

# Fast model used for the intent classifier.  We only need a yes/no answer
# so the smallest/cheapest available model is best.
_INTENT_CLASSIFIER_MODEL = "gemini-3.1-flash-lite"

# One-line summary of each tool extracted from _TOOLS, injected into the
# classifier prompt so it only returns YES when the intent matches something
# the model can actually do.  Built once at module load time.
def _build_tool_summary() -> str:
    lines = []
    for group in _TOOLS:
        for decl in group.get("function_declarations", []):
            name = decl.get("name", "")
            desc = decl.get("description", "")
            first_sentence = desc.split(".")[0].strip()
            lines.append(f"- {name}: {first_sentence}")
    return "\n".join(lines)

_TOOL_SUMMARY = _build_tool_summary()

_INTENT_CLASSIFIER_PROMPT = """\
You are a classifier for an AI voice assistant.

The assistant has access to these tools:
{tools}

The assistant said the following to a user (this is a transcript of its speech):

"{output}"

Did the assistant announce or imply that it was ABOUT TO use one of the above tools \
(e.g. search, open a dialog, generate an image, look something up, check what is \
selected) but its response ended WITHOUT the tool actually being called?

Only answer YES if:
1. The assistant clearly expressed intent to perform an action that matches one of the \
tools listed above.
2. The response ended without that action being completed (i.e. the assistant said \
it would do something but stopped short of doing it).

Answer NO if:
- The response was just a normal conversational reply.
- The assistant already reported a completed result ("I found...", "Here are...", etc.).
- The assistant asked the user a question without announcing a tool action.
- The intent does not match any of the listed tools.

Answer with exactly one word: YES or NO.
"""


async def _classify_missed_tool_call(output: str, client: object) -> bool:
    """
    Use a lightweight Gemini call to decide whether the model announced a tool
    action but completed the turn without calling it.

    Works in any language the model might be speaking (English, Arabic, etc.).
    Returns False on any error so the recovery is only triggered when confident.
    """
    words = output.split()
    if len(words) < _MIN_WORDS_FOR_INTENT_CHECK:
        return False  # too short to be a tool-intent announcement

    prompt = _INTENT_CLASSIFIER_PROMPT.format(tools=_TOOL_SUMMARY, output=output.strip())
    try:
        resp = await client.aio.models.generate_content(  # type: ignore[attr-defined]
            model=_INTENT_CLASSIFIER_MODEL,
            contents=prompt,
        )
        answer = (resp.text or "").strip().upper()
        _dbg(f"[LIVE 🔍 INTENT] classifier → {answer!r}  (input: {output[:80]!r}...)")
        return answer.startswith("YES")
    except Exception as exc:
        logger.warning("Intent classifier error (skipping retry): %s", exc)
        return False


def _summarise_message(msg: object) -> str:
    """Return a compact one-line description of a LiveServerMessage for logging."""
    parts = []
    sc = getattr(msg, "server_content", None)
    if sc:
        if getattr(sc, "turn_complete", False):
            parts.append("turn_complete")
        if getattr(sc, "interrupted", False):
            parts.append("interrupted")
        mt = getattr(sc, "model_turn", None)
        if mt:
            n_audio = sum(
                1 for p in (getattr(mt, "parts", None) or [])
                if getattr(p, "inline_data", None)
            )
            n_text = sum(
                1 for p in (getattr(mt, "parts", None) or [])
                if getattr(p, "text", None)
            )
            if n_audio:
                parts.append(f"audio×{n_audio}")
            if n_text:
                parts.append(f"text×{n_text}")
        if getattr(sc, "input_transcription", None):
            parts.append("input_tx")
        if getattr(sc, "output_transcription", None):
            parts.append("output_tx")
    if getattr(msg, "session_resumption_update", None):
        parts.append("resumption_update")
    if getattr(msg, "go_away", None) is not None:
        parts.append("go_away")
    if getattr(msg, "usage_metadata", None):
        parts.append("usage_metadata")
    tc = getattr(msg, "tool_call", None)
    if tc:
        fcs = getattr(tc, "function_calls", None) or []
        names = [getattr(f, "name", "?") for f in fcs]
        parts.append(f"tool_call({', '.join(names)})")
    return ", ".join(parts) if parts else repr(msg)[:80]
