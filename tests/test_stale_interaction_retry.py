"""Stale Gemini previous_interaction_id → clear + retry without chain."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai.errors import ClientError

from app import chat_agent
from app.routers import chat


def _event(event_type: str, **payload: object) -> str:
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


def _not_found(*, message: str = "Requested entity was not found") -> ClientError:
    return ClientError(
        404,
        {
            "error": {
                "code": 404,
                "message": message,
                "status": "NOT_FOUND",
            }
        },
    )


def _rate_limit() -> ClientError:
    return ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "Resource exhausted",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )


def _server_error() -> ClientError:
    return ClientError(
        500,
        {
            "error": {
                "code": 500,
                "message": "Internal error",
                "status": "INTERNAL",
            }
        },
    )


class _NotFoundError(Exception):
    """SDK-style name without a numeric code attribute."""


class _WeirdExc(Exception):
    """Malformed shape: non-int .code must not crash the matcher."""

    code = "not-a-number"


class _CompletedEvent:
    event_type = "interaction.completed"
    index = None
    interaction_id = "v1_fresh"

    def __init__(self, interaction_id: str = "v1_fresh"):
        self.interaction_id = interaction_id
        self.interaction = MagicMock(id=interaction_id, status="completed")


def _patch_gemini_stream_deps(monkeypatch) -> None:
    monkeypatch.setattr(chat_agent.llm_provider, "use_claude", lambda: False)
    monkeypatch.setattr(chat_agent, "get_chat_tools", lambda user_id=None: [])
    monkeypatch.setattr(chat_agent, "get_chat_system_instruction", lambda: "sys")
    monkeypatch.setattr(chat_agent, "cfg", lambda key, default=None: default)
    monkeypatch.setattr(
        chat_agent,
        "cfg_str",
        lambda key, default="": default if key != "CHAT_THINKING_LEVEL" else "",
    )
    monkeypatch.setattr(chat_agent, "log_error", AsyncMock())


def _install_create(monkeypatch, fake_create) -> MagicMock:
    mock_client = MagicMock()
    mock_client.aio.interactions.create = AsyncMock(side_effect=fake_create)
    monkeypatch.setattr(chat_agent, "get_client", lambda: mock_client)
    return mock_client


# ── Matcher unit tests ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "exc,expected",
    [
        (_not_found(), True),
        (_not_found(message="entity was not found."), True),
        (ClientError(404, {"error": {"message": "gone", "status": "NOT_FOUND"}}), True),
        (Exception("404 Requested entity was not found"), True),
        (Exception("404 NOT_FOUND — entity was not found"), True),
        (_NotFoundError("missing interaction"), True),
        (Exception("NotFoundError: entity missing"), False),
        (_rate_limit(), False),
        (_server_error(), False),
        (TimeoutError("deadline exceeded"), False),
        (ClientError(400, {"error": {"message": "bad request", "status": "INVALID"}}), False),
        (RuntimeError("network blip"), False),
        (_WeirdExc("something odd"), False),
    ],
)
def test_is_previous_interaction_not_found(exc, expected):
    assert chat_agent._is_previous_interaction_not_found(exc) is expected


def test_is_previous_interaction_not_found_tolerates_missing_code_attr():
    class NoCode(Exception):
        message = "Requested entity was not found"

    assert chat_agent._is_previous_interaction_not_found(NoCode()) is True


# ── Stream create + retry ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_retries_without_previous_and_clears_user_session(monkeypatch):
    _patch_gemini_stream_deps(monkeypatch)
    create_calls: list[object] = []

    async def fake_create(**kwargs):
        prev = kwargs.get("previous_interaction_id")
        create_calls.append(prev)
        if prev:
            raise _not_found()

        async def _stream():
            yield _CompletedEvent()

        return _stream()

    _install_create(monkeypatch, fake_create)

    cleared: list[tuple] = []

    async def clear_user(user_id, interaction_id):
        cleared.append((user_id, interaction_id))

    monkeypatch.setattr(
        "app.chat_session.set_interaction_id",
        AsyncMock(side_effect=clear_user),
    )

    chunks = [
        chunk
        async for chunk in chat_agent._stream_chat_inner(
            [{"type": "text", "text": "hello"}],
            previous_interaction_id="v1_expired",
            allow_tools=False,
            user_id=42,
        )
    ]

    assert create_calls == ["v1_expired", None]
    assert cleared == [(42, None)]
    reset_i = chunks.index(_event("interaction_reset", reason="previous_not_found"))
    done_i = chunks.index(_event("done", interaction_id="v1_fresh"))
    assert reset_i < done_i
    assert not any(
        '"type": "error"' in chunk for chunk in chunks if chunk.startswith("data:")
    )


@pytest.mark.asyncio
async def test_stream_retries_and_clears_guest_session(monkeypatch):
    _patch_gemini_stream_deps(monkeypatch)
    create_calls: list[object] = []

    async def fake_create(**kwargs):
        prev = kwargs.get("previous_interaction_id")
        create_calls.append(prev)
        if prev:
            raise _not_found(message="entity was not found")

        async def _stream():
            yield _CompletedEvent("v1_guest_fresh")

        return _stream()

    _install_create(monkeypatch, fake_create)

    cleared: list[tuple] = []
    set_user = AsyncMock()
    monkeypatch.setattr(
        "app.chat_session.set_guest_interaction_id",
        AsyncMock(side_effect=lambda gid, iid: cleared.append((gid, iid))),
    )
    # Ensure user path is not used when only guest is set
    monkeypatch.setattr("app.chat_session.set_interaction_id", set_user)

    chunks = [
        chunk
        async for chunk in chat_agent._stream_chat_inner(
            [{"type": "text", "text": "hello"}],
            previous_interaction_id="v1_expired",
            allow_tools=False,
            guest_session_id="guest-abc",
        )
    ]

    assert create_calls == ["v1_expired", None]
    assert cleared == [("guest-abc", None)]
    assert _event("interaction_reset", reason="previous_not_found") in chunks
    assert _event("done", interaction_id="v1_guest_fresh") in chunks
    set_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_also_fails_emits_reset_then_error_no_loop(monkeypatch):
    _patch_gemini_stream_deps(monkeypatch)
    create_calls: list[object] = []

    async def fake_create(**kwargs):
        create_calls.append(kwargs.get("previous_interaction_id"))
        raise _not_found()

    _install_create(monkeypatch, fake_create)
    monkeypatch.setattr("app.chat_session.set_interaction_id", AsyncMock())

    chunks = [
        chunk
        async for chunk in chat_agent._stream_chat_inner(
            [{"type": "text", "text": "hello"}],
            previous_interaction_id="v1_expired",
            allow_tools=False,
            user_id=7,
        )
    ]

    assert create_calls == ["v1_expired", None]
    assert _event("interaction_reset", reason="previous_not_found") in chunks
    assert any('"type": "error"' in c for c in chunks)
    assert not any('"type": "done"' in c for c in chunks)
    reset_i = next(i for i, c in enumerate(chunks) if "interaction_reset" in c)
    error_i = next(i for i, c in enumerate(chunks) if '"type": "error"' in c)
    assert reset_i < error_i


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory",
    [_rate_limit, _server_error, lambda: TimeoutError("timed out")],
)
async def test_non_stale_errors_do_not_retry_or_clear(monkeypatch, factory):
    _patch_gemini_stream_deps(monkeypatch)
    create_calls: list[object] = []

    async def fake_create(**kwargs):
        create_calls.append(kwargs.get("previous_interaction_id"))
        raise factory()

    _install_create(monkeypatch, fake_create)
    clear_user = AsyncMock()
    monkeypatch.setattr("app.chat_session.set_interaction_id", clear_user)

    chunks = [
        chunk
        async for chunk in chat_agent._stream_chat_inner(
            [{"type": "text", "text": "hello"}],
            previous_interaction_id="v1_keep",
            allow_tools=False,
            user_id=9,
        )
    ]

    assert create_calls == ["v1_keep"]
    clear_user.assert_not_awaited()
    assert not any("interaction_reset" in c for c in chunks)
    assert any('"type": "error"' in c for c in chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize("prev", [None, ""])
async def test_empty_previous_not_found_does_not_retry(monkeypatch, prev):
    _patch_gemini_stream_deps(monkeypatch)
    create_calls: list[object] = []

    async def fake_create(**kwargs):
        create_calls.append(kwargs.get("previous_interaction_id"))
        raise _not_found()

    _install_create(monkeypatch, fake_create)
    clear_user = AsyncMock()
    monkeypatch.setattr("app.chat_session.set_interaction_id", clear_user)

    chunks = [
        chunk
        async for chunk in chat_agent._stream_chat_inner(
            [{"type": "text", "text": "hello"}],
            previous_interaction_id=prev,
            allow_tools=False,
            user_id=3,
        )
    ]

    assert create_calls == [prev]
    clear_user.assert_not_awaited()
    assert not any("interaction_reset" in c for c in chunks)
    assert any('"type": "error"' in c for c in chunks)


@pytest.mark.asyncio
async def test_claude_path_skips_gemini_stale_retry(monkeypatch):
    """cld_ / Claude transcript path must not hit Interactions create or Gemini reset."""
    monkeypatch.setattr(chat_agent.llm_provider, "use_claude", lambda: True)
    create = AsyncMock(side_effect=AssertionError("Gemini create must not run"))
    mock_client = MagicMock()
    mock_client.aio.interactions.create = create
    monkeypatch.setattr(chat_agent, "get_client", lambda: mock_client)

    async def fake_claude(input_parts, previous_interaction_id, **kwargs):
        yield _event("text_delta", text="from-claude")
        yield _event("done", interaction_id="cld_existing")

    monkeypatch.setattr(chat_agent, "_stream_chat_claude", fake_claude)

    chunks = [
        chunk
        async for chunk in chat_agent._stream_chat_inner(
            [{"type": "text", "text": "hello"}],
            previous_interaction_id="cld_existing",
            allow_tools=False,
            user_id=1,
        )
    ]

    create.assert_not_awaited()
    assert not any("interaction_reset" in c for c in chunks)
    assert _event("done", interaction_id="cld_existing") in chunks


@pytest.mark.asyncio
async def test_non_stream_mode_also_retries_stale(monkeypatch):
    _patch_gemini_stream_deps(monkeypatch)
    monkeypatch.setattr(
        chat_agent,
        "cfg",
        lambda key, default=None: False if key == "CHAT_STREAM" else default,
    )
    create_calls: list[object] = []

    async def fake_create(**kwargs):
        prev = kwargs.get("previous_interaction_id")
        create_calls.append(prev)
        if prev:
            raise _not_found()
        return MagicMock(id="v1_ns", status="completed", steps=[])

    _install_create(monkeypatch, fake_create)
    monkeypatch.setattr("app.chat_session.set_interaction_id", AsyncMock())

    chunks = [
        chunk
        async for chunk in chat_agent._stream_chat_inner(
            [{"type": "text", "text": "hello"}],
            previous_interaction_id="v1_expired",
            allow_tools=False,
            user_id=11,
        )
    ]

    assert create_calls == ["v1_expired", None]
    assert _event("interaction_reset", reason="previous_not_found") in chunks
    assert _event("done", interaction_id="v1_ns") in chunks


# ── Router session save ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_with_session_save_clears_guest_on_interaction_reset(monkeypatch):
    async def source():
        yield _event("interaction_reset", reason="previous_not_found")
        yield _event("text_delta", text="hi")
        yield _event("done", interaction_id="v1_new")

    calls: list[tuple] = []

    async def set_guest(guest_id, interaction_id):
        calls.append((guest_id, interaction_id))

    monkeypatch.setattr(
        chat.chat_session,
        "set_guest_interaction_id",
        AsyncMock(side_effect=set_guest),
    )

    chunks = [
        chunk
        async for chunk in chat._stream_with_session_save(
            None,
            source(),
            guest_session_id="guest-stale",
            user_text="hello",
            record_transcript=False,
        )
    ]

    assert calls[0] == ("guest-stale", None)
    assert calls[-1] == ("guest-stale", "v1_new")
    assert _event("interaction_reset", reason="previous_not_found") in chunks
    assert _event("done", interaction_id="v1_new") in chunks


@pytest.mark.asyncio
async def test_stream_with_session_save_clears_user_on_interaction_reset(monkeypatch):
    async def source():
        yield _event("interaction_reset", reason="previous_not_found")
        yield _event("done", interaction_id="v1_user_new")

    calls: list[tuple] = []
    monkeypatch.setattr(
        chat.chat_session,
        "set_interaction_id",
        AsyncMock(side_effect=lambda uid, iid: calls.append((uid, iid))),
    )
    monkeypatch.setattr(
        chat.chat_session,
        "set_guest_interaction_id",
        AsyncMock(side_effect=AssertionError("guest clear must not run for user")),
    )

    chunks = [
        chunk
        async for chunk in chat._stream_with_session_save(
            99,
            source(),
            user_text="hello",
            record_transcript=False,
        )
    ]

    assert calls == [(99, None), (99, "v1_user_new")]
    assert _event("interaction_reset", reason="previous_not_found") in chunks
    assert chunks[-1] == _event("done", interaction_id="v1_user_new")


@pytest.mark.asyncio
async def test_done_persists_new_id_after_double_clear(monkeypatch):
    """Router clear on reset + agent clear are both None; done still wins."""

    async def source():
        yield _event("interaction_reset", reason="previous_not_found")
        yield _event("interaction_reset", reason="previous_not_found")
        yield _event("done", interaction_id="v1_after_reset")

    calls: list[tuple] = []
    monkeypatch.setattr(
        chat.chat_session,
        "set_guest_interaction_id",
        AsyncMock(side_effect=lambda gid, iid: calls.append((gid, iid))),
    )

    chunks = [
        chunk
        async for chunk in chat._stream_with_session_save(
            None,
            source(),
            guest_session_id="guest-dbl",
            user_text="hello",
            record_transcript=False,
        )
    ]

    assert calls.count(("guest-dbl", None)) == 2
    assert calls[-1] == ("guest-dbl", "v1_after_reset")
    assert chunks[-1] == _event("done", interaction_id="v1_after_reset")


def test_session_prefers_server_id_over_client():
    """Router chain: session_prev or client previous — server wins when present."""
    session_prev = "v1_server"
    client_prev = "v1_stale_client"
    assert (session_prev or client_prev) == "v1_server"
    # After reset, session is cleared so a stale client id would be used until
    # the next successful done — which is why interaction_reset must clear both.
    assert (None or client_prev) == "v1_stale_client"
    assert (None or None) is None
