"""Voice go_away should not be forwarded as a client-teardown event."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.live_session import GeminiLiveSession, LiveEventType


@pytest.mark.asyncio
async def test_go_away_emits_reconnecting_not_go_away():
    session = GeminiLiveSession.__new__(GeminiLiveSession)
    session.user_id = 1
    session.guest_session_id = None
    session.resumption_handle = "handle-abc"
    session._event_queue = AsyncMock()
    session._event_queue.put = AsyncMock()
    # Fields touched later in _dispatch for other message parts
    session._pending_audio_bytes = 0
    session.model = "test-model"

    go_away = SimpleNamespace(time_left=SimpleNamespace(total_seconds=lambda: 12.0))
    message = SimpleNamespace(
        go_away=go_away,
        session_resumption_update=None,
        usage_metadata=None,
        server_content=None,
        tool_call=None,
        tool_call_cancellation=None,
    )

    await session._dispatch(message)

    assert session._event_queue.put.await_count >= 1
    event = session._event_queue.put.await_args.args[0]
    assert event.type == LiveEventType.RECONNECTING
    assert event.type != LiveEventType.GO_AWAY
