import json
from unittest.mock import AsyncMock

import pytest

from app.routers import chat


def _event(event_type: str, **payload: object) -> str:
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


@pytest.mark.asyncio
async def test_done_is_delivered_when_final_persistence_fails(monkeypatch):
    async def source():
        yield _event("text_delta", text="Completed answer")
        yield _event("done", interaction_id="v1_test")

    persist = AsyncMock(side_effect=RuntimeError("database unavailable"))
    log_error = AsyncMock()
    monkeypatch.setattr(chat.chat_session, "set_guest_interaction_id", persist)
    monkeypatch.setattr(chat, "log_error", log_error)

    chunks = [
        chunk
        async for chunk in chat._stream_with_session_save(
            None,
            source(),
            guest_session_id="guest-test",
            user_text="hello",
        )
    ]

    assert chunks == [
        _event("text_delta", text="Completed answer"),
        _event("done", interaction_id="v1_test"),
    ]
    persist.assert_awaited_once_with("guest-test", "v1_test")
    log_error.assert_awaited_once()

