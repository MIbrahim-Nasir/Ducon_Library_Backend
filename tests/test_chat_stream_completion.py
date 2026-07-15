import asyncio
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

    assert ": keepalive\n\n" in chunks
    assert chunks[-2:] == [
        _event("text_delta", text="Completed answer"),
        _event("done", interaction_id="v1_test"),
    ]
    persist.assert_awaited_once_with("guest-test", "v1_test")
    log_error.assert_awaited_once()


@pytest.mark.asyncio
async def test_keepalive_does_not_cancel_slow_model_stream():
    """Idle gaps must emit keepalives without cancelling the producer.

    The old wait_for(aiter.__anext__) pattern cancelled Gemini mid-turn after
    15s of silence, which surfaced as 'SSE closed without done event'.
    """

    async def source():
        await asyncio.sleep(0.25)
        yield _event("text_delta", text="late")
        yield _event("done", interaction_id="v1_slow")

    chunks = [
        chunk
        async for chunk in chat._stream_with_session_save(
            None,
            source(),
            guest_session_id="guest-slow",
            user_text="hello",
            record_transcript=False,
            keepalive_interval=0.05,
        )
    ]

    assert ": keepalive\n\n" in chunks
    assert _event("text_delta", text="late") in chunks
    assert _event("done", interaction_id="v1_slow") in chunks
    assert chunks[-1] == _event("done", interaction_id="v1_slow")

