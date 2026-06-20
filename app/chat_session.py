"""
Per-user chat session store — tracks the Gemini Interactions API interaction_id
so voice delegation and text chat share one conversation chain.
"""
from __future__ import annotations

from typing import Optional

# user_id → interaction_id (in-memory; survives across voice/chat within server process)
_sessions: dict[int, str] = {}


def get_interaction_id(user_id: int) -> Optional[str]:
    return _sessions.get(user_id)


def set_interaction_id(user_id: int, interaction_id: Optional[str]) -> None:
    if interaction_id:
        _sessions[user_id] = interaction_id
    elif user_id in _sessions:
        del _sessions[user_id]


def clear_session(user_id: int) -> None:
    _sessions.pop(user_id, None)
