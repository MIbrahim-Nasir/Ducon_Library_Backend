"""
Per-user chat session store — tracks the Gemini Interactions API interaction_id
and a compact text transcript so voice and text chat share one conversation chain.

Authenticated users are keyed by user_id; guests by guest session UUID string.

Voice → chat: completed voice turns are injected via /chat/voice_context (Interactions API).
Chat → voice: the transcript accumulated here is seeded into Gemini Live on connect.
"""
from __future__ import annotations

from typing import Optional

# user_id → interaction_id (in-memory; survives across voice/chat within server process)
_sessions: dict[int, str] = {}
_guest_sessions: dict[str, str] = {}

# Compact transcript for Live session seeding
_transcripts: dict[int, list[dict[str, str]]] = {}
_guest_transcripts: dict[str, list[dict[str, str]]] = {}

_MAX_TRANSCRIPT_TURNS = 40


def get_interaction_id(user_id: int) -> Optional[str]:
    return _sessions.get(user_id)


def set_interaction_id(user_id: int, interaction_id: Optional[str]) -> None:
    if interaction_id:
        _sessions[user_id] = interaction_id
    elif user_id in _sessions:
        del _sessions[user_id]


def get_guest_interaction_id(guest_session_id: str) -> Optional[str]:
    return _guest_sessions.get(guest_session_id)


def set_guest_interaction_id(guest_session_id: str, interaction_id: Optional[str]) -> None:
    if interaction_id:
        _guest_sessions[guest_session_id] = interaction_id
    elif guest_session_id in _guest_sessions:
        del _guest_sessions[guest_session_id]


def append_turn(user_id: int, user_text: str, model_text: str) -> None:
    """Record one completed chat/voice-inject turn for Live session seeding."""
    turns = _transcripts.setdefault(user_id, [])
    user_clean = (user_text or "").strip()
    model_clean = (model_text or "").strip()
    if user_clean:
        turns.append({"role": "user", "text": user_clean})
    if model_clean:
        turns.append({"role": "model", "text": model_clean})
    if len(turns) > _MAX_TRANSCRIPT_TURNS:
        _transcripts[user_id] = turns[-_MAX_TRANSCRIPT_TURNS:]


def append_guest_turn(guest_session_id: str, user_text: str, model_text: str) -> None:
    turns = _guest_transcripts.setdefault(guest_session_id, [])
    user_clean = (user_text or "").strip()
    model_clean = (model_text or "").strip()
    if user_clean:
        turns.append({"role": "user", "text": user_clean})
    if model_clean:
        turns.append({"role": "model", "text": model_clean})
    if len(turns) > _MAX_TRANSCRIPT_TURNS:
        _guest_transcripts[guest_session_id] = turns[-_MAX_TRANSCRIPT_TURNS:]


def _to_live_contents(raw: list[dict[str, str]]) -> list[dict]:
    seeded: list[dict] = []
    for turn in raw:
        text = (turn.get("text") or "").strip()
        role = turn.get("role")
        if not text or role not in ("user", "model"):
            continue
        seeded.append({"role": role, "parts": [{"text": text}]})
    return seeded[-_MAX_TRANSCRIPT_TURNS:]


def get_voice_seed_turns(user_id: int) -> list[dict]:
    """Return turns in Gemini Live Content dict format."""
    return _to_live_contents(_transcripts.get(user_id) or [])


def get_guest_voice_seed_turns(guest_session_id: str) -> list[dict]:
    return _to_live_contents(_guest_transcripts.get(guest_session_id) or [])


def clear_session(user_id: int) -> None:
    _sessions.pop(user_id, None)
    _transcripts.pop(user_id, None)


def clear_guest_session(guest_session_id: str) -> None:
    _guest_sessions.pop(guest_session_id, None)
    _guest_transcripts.pop(guest_session_id, None)
