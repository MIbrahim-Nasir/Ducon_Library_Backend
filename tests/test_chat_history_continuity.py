"""Follow-up turns must keep prior user text + not greet on a leftover '?'."""
from __future__ import annotations

import pytest

from app import chat_agent
from app import llm_provider


def test_resolve_chain_prefers_fresh_client_id_over_stale_server():
    """Client just got `done`; Postgres may still hold the previous id."""
    assert (
        chat_agent.resolve_chain_previous_id(
            "v1_old", "v1_new", use_claude=False
        )
        == "v1_new"
    )
    assert (
        chat_agent.resolve_chain_previous_id(
            "cld_old", "cld_new", use_claude=True
        )
        == "cld_new"
    )


def test_resolve_chain_uses_server_when_client_has_none():
    assert (
        chat_agent.resolve_chain_previous_id("v1_server", None, use_claude=False)
        == "v1_server"
    )
    assert (
        chat_agent.resolve_chain_previous_id("cld_server", None, use_claude=True)
        == "cld_server"
    )


def test_resolve_chain_message_turn_does_not_inherit_server_id():
    """Empty chat UI omits client id — must not resume the stored thread."""
    assert (
        chat_agent.resolve_chain_previous_id(
            "v1_server", None, use_claude=False, allow_session_fallback=False
        )
        is None
    )
    assert (
        chat_agent.resolve_chain_previous_id(
            "v1_server", "v1_client", use_claude=False, allow_session_fallback=False
        )
        == "v1_client"
    )


def test_resolve_chain_does_not_mix_gemini_and_claude_ids():
    assert (
        chat_agent.resolve_chain_previous_id("v1_gemini", None, use_claude=True)
        is None
    )
    assert (
        chat_agent.resolve_chain_previous_id(None, "v1_gemini", use_claude=True)
        is None
    )
    assert (
        chat_agent.resolve_chain_previous_id("cld_claude", "v1_gemini", use_claude=True)
        == "cld_claude"
    )


def test_seed_turns_to_claude_messages_keeps_prior_user_turn():
    turns = [
        {"role": "user", "parts": [{"text": "the benches and fountains are ducon products used."}]},
        {"role": "model", "parts": [{"text": "Starting a design run…"}]},
    ]
    messages = chat_agent.seed_turns_to_claude_messages(turns)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert "benches and fountains" in messages[0]["content"][0]["text"]
    assert messages[1]["content"][0]["text"].startswith("Starting")


def test_follow_up_question_mark_still_sees_prior_design_request():
    """The screenshot case: leftover '?' must not be the only user turn."""
    prior = chat_agent.seed_turns_to_claude_messages([
        {
            "role": "user",
            "parts": [{
                "text": (
                    "the benches and fountains are ducon products used. "
                    "make the same fountain pushed back. use autonomous designer job"
                ),
            }],
        },
        {"role": "model", "parts": [{"text": "I'll integrate the fountain with the bench."}]},
    ])
    follow_up = chat_agent._build_claude_user_message(
        [{"type": "text", "text": "?"}],
        prior,
    )
    messages = [*prior, follow_up]
    user_texts = [
        block["text"]
        for msg in messages
        if msg["role"] == "user"
        for block in msg["content"]
        if block.get("type") == "text"
    ]
    assert any("autonomous designer" in t for t in user_texts)
    assert user_texts[-1] == "?"
    assert len(messages) == 3


@pytest.mark.asyncio
async def test_claude_hydrates_empty_worker_memory_from_transcript(monkeypatch):
    """Worker B has no _CLAUDE_HISTORY — load Postgres transcript before the turn."""
    chat_agent._CLAUDE_HISTORY.clear()

    async def fake_seed(user_id):
        assert user_id == 42
        return [
            {"role": "user", "parts": [{"text": "keep the fountain materials"}]},
            {"role": "model", "parts": [{"text": "Noted the materials."}]},
        ]

    captured: dict = {}

    class _FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def __aiter__(self):
            if False:
                yield None

        async def get_final_message(self):
            return type("Msg", (), {"content": []})()

    class _FakeClient:
        class messages:
            @staticmethod
            def stream(**kwargs):
                captured["messages"] = kwargs["messages"]
                return _FakeStream()

    monkeypatch.setattr(
        "app.chat_session.get_voice_seed_turns",
        fake_seed,
    )
    monkeypatch.setattr(llm_provider, "get_async_anthropic_client", lambda: _FakeClient())
    monkeypatch.setattr(llm_provider, "_thinking_param", lambda: None)
    monkeypatch.setattr(llm_provider, "serialize_content", lambda _m: [])
    monkeypatch.setattr(llm_provider, "tool_use_blocks", lambda _m: [])
    monkeypatch.setattr(chat_agent, "get_chat_system_instruction", lambda: "sys")
    monkeypatch.setattr(chat_agent, "get_claude_chat_tools", lambda **_k: [])

    chunks = [
        chunk
        async for chunk in chat_agent._stream_chat_claude(
            [{"type": "text", "text": "now push the fountain back"}],
            "cld_missing_on_this_worker",
            user_id=42,
        )
    ]

    roles = [m["role"] for m in captured["messages"]]
    texts = [
        b["text"]
        for m in captured["messages"]
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    assert roles[0] == "user"
    assert "keep the fountain materials" in texts[0]
    assert texts[-1] == "now push the fountain back"
    assert any("done" in c for c in chunks)


@pytest.mark.asyncio
async def test_claude_new_session_does_not_hydrate_transcript(monkeypatch):
    chat_agent._CLAUDE_HISTORY.clear()

    async def fake_seed(user_id):
        raise AssertionError("new session must not load prior transcript")

    captured: dict = {}

    class _FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def __aiter__(self):
            if False:
                yield None

        async def get_final_message(self):
            return type("Msg", (), {"content": []})()

    class _FakeClient:
        class messages:
            @staticmethod
            def stream(**kwargs):
                captured["messages"] = kwargs["messages"]
                return _FakeStream()

    monkeypatch.setattr("app.chat_session.get_voice_seed_turns", fake_seed)
    monkeypatch.setattr(llm_provider, "get_async_anthropic_client", lambda: _FakeClient())
    monkeypatch.setattr(llm_provider, "_thinking_param", lambda: None)
    monkeypatch.setattr(llm_provider, "serialize_content", lambda _m: [])
    monkeypatch.setattr(llm_provider, "tool_use_blocks", lambda _m: [])
    monkeypatch.setattr(chat_agent, "get_chat_system_instruction", lambda: "sys")
    monkeypatch.setattr(chat_agent, "get_claude_chat_tools", lambda **_k: [])

    async for _ in chat_agent._stream_chat_claude(
        [{"type": "text", "text": "hello?"}],
        None,
        user_id=42,
    ):
        pass

    texts = [
        b["text"]
        for m in captured["messages"]
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    assert texts == ["hello?"]


def test_transcript_turns_to_gemini_prefix_keeps_design_request():
    prefix = chat_agent.transcript_turns_to_gemini_prefix([
        {
            "role": "user",
            "parts": [{
                "text": (
                    "the benches and fountains are ducon products used. "
                    "make the same fountain pushed back"
                ),
            }],
        },
        {"role": "model", "parts": [{"text": "I'll integrate the fountain."}]},
    ])
    assert len(prefix) == 1
    text = prefix[0]["text"]
    assert "do not greet as a new chat" in text
    assert "benches and fountains" in text
    assert "I'll integrate the fountain." in text


@pytest.mark.asyncio
async def test_gemini_stale_retry_prepends_transcript(monkeypatch):
    """Expired previous_interaction_id must not wipe the design request."""
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setattr(llm_provider, "use_claude", lambda: False)
    monkeypatch.setattr(chat_agent, "get_chat_tools", lambda user_id=None: [])
    monkeypatch.setattr(chat_agent, "get_chat_system_instruction", lambda: "sys")
    monkeypatch.setattr(chat_agent, "cfg", lambda key, default=None: default)
    monkeypatch.setattr(
        chat_agent,
        "cfg_str",
        lambda key, default="": default if key != "CHAT_THINKING_LEVEL" else "",
    )
    monkeypatch.setattr(chat_agent, "log_error", AsyncMock())

    async def fake_seed(user_id):
        assert user_id == 7
        return [
            {
                "role": "user",
                "parts": [{"text": "the benches and fountains are ducon products used."}],
            },
            {"role": "model", "parts": [{"text": "Starting a design run."}]},
        ]

    monkeypatch.setattr("app.chat_session.get_voice_seed_turns", fake_seed)
    monkeypatch.setattr("app.chat_session.set_interaction_id", AsyncMock())

    captured: list[list] = []

    class _Done:
        event_type = "interaction.completed"
        index = None
        interaction_id = "v1_fresh"
        interaction = MagicMock(id="v1_fresh", status="completed")

    async def fake_create(**kwargs):
        captured.append(list(kwargs.get("input") or []))
        if kwargs.get("previous_interaction_id"):
            raise Exception("404 Requested entity was not found")

        async def _stream():
            yield _Done()

        return _stream()

    mock_client = MagicMock()
    mock_client.aio.interactions.create = AsyncMock(side_effect=fake_create)
    monkeypatch.setattr(chat_agent, "get_client", lambda: mock_client)

    chunks = [
        chunk
        async for chunk in chat_agent._stream_chat_inner(
            [{"type": "text", "text": "now push the fountain back"}],
            previous_interaction_id="v1_expired",
            allow_tools=False,
            user_id=7,
        )
    ]

    assert len(captured) == 2
    assert captured[0][0]["text"] == "now push the fountain back"
    retry_text = captured[1][0]["text"]
    assert "benches and fountains" in retry_text
    assert "do not greet as a new chat" in retry_text
    assert captured[1][-1]["text"] == "now push the fountain back"
    assert any("interaction_reset" in c for c in chunks)


@pytest.mark.asyncio
async def test_gemini_no_chain_does_not_hydrate_prior_session(monkeypatch):
    """Empty UI / no client id must not inherit yesterday's transcript."""
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setattr(llm_provider, "use_claude", lambda: False)
    monkeypatch.setattr(chat_agent, "get_chat_tools", lambda user_id=None: [])
    monkeypatch.setattr(chat_agent, "get_chat_system_instruction", lambda: "sys")
    monkeypatch.setattr(chat_agent, "cfg", lambda key, default=None: default)
    monkeypatch.setattr(
        chat_agent,
        "cfg_str",
        lambda key, default="": default if key != "CHAT_THINKING_LEVEL" else "",
    )
    monkeypatch.setattr(chat_agent, "log_error", AsyncMock())
    seed = AsyncMock(return_value=[
        {"role": "user", "parts": [{"text": "show me pergola options"}]},
        {"role": "model", "parts": [{"text": "Here are pergolas."}]},
    ])
    monkeypatch.setattr("app.chat_session.get_voice_seed_turns", seed)

    captured: dict = {}

    class _Done:
        event_type = "interaction.completed"
        index = None
        interaction_id = "v1_new"
        interaction = MagicMock(id="v1_new", status="completed")

    async def fake_create(**kwargs):
        captured["input"] = list(kwargs.get("input") or [])

        async def _stream():
            yield _Done()

        return _stream()

    mock_client = MagicMock()
    mock_client.aio.interactions.create = AsyncMock(side_effect=fake_create)
    monkeypatch.setattr(chat_agent, "get_client", lambda: mock_client)

    async for _ in chat_agent._stream_chat_inner(
        [{"type": "text", "text": "hello?"}],
        previous_interaction_id=None,
        allow_tools=False,
        user_id=9,
    ):
        pass

    assert captured["input"][-1]["text"] == "hello?"
    assert all("pergola" not in (p.get("text") or "") for p in captured["input"])
    seed.assert_not_awaited()


@pytest.mark.asyncio
async def test_gemini_stream_error_emits_sse_error_without_done(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setattr(llm_provider, "use_claude", lambda: False)
    monkeypatch.setattr(chat_agent, "get_chat_tools", lambda user_id=None: [])
    monkeypatch.setattr(chat_agent, "get_chat_system_instruction", lambda: "sys")
    monkeypatch.setattr(chat_agent, "cfg", lambda key, default=None: default)
    monkeypatch.setattr(
        chat_agent,
        "cfg_str",
        lambda key, default="": default if key != "CHAT_THINKING_LEVEL" else "",
    )
    monkeypatch.setattr(chat_agent, "log_error", AsyncMock())
    clear_id = AsyncMock()
    monkeypatch.setattr(chat_agent, "_clear_stored_interaction_id", clear_id)

    class _Err:
        event_type = "error"
        index = None
        interaction_id = "v1_failed"
        error = MagicMock(message="INTERNAL")
        interaction = MagicMock(id="v1_failed")

    async def fake_create(**kwargs):
        async def _stream():
            yield _Err()

        return _stream()

    mock_client = MagicMock()
    mock_client.aio.interactions.create = AsyncMock(side_effect=fake_create)
    monkeypatch.setattr(chat_agent, "get_client", lambda: mock_client)

    chunks = [
        chunk
        async for chunk in chat_agent._stream_chat_inner(
            [{"type": "text", "text": "integrate the fountain"}],
            previous_interaction_id="v1_prev",
            allow_tools=False,
            user_id=10,
        )
    ]

    assert any('"type": "error"' in c and "INTERNAL" in c for c in chunks)
    assert not any('"type": "done"' in c for c in chunks)
    clear_id.assert_awaited()


def test_gemini_media_type_from_mime():
    assert chat_agent.gemini_media_type_from_mime("image/png") == "image"
    assert chat_agent.gemini_media_type_from_mime("application/pdf") == "document"
    assert chat_agent.gemini_media_type_from_mime("video/mp4") == "video"


@pytest.mark.asyncio
async def test_build_gemini_media_part_inlines_small_images(monkeypatch):
    async def fake_upload(**kwargs):
        raise AssertionError("small chat images must not use the Files API")

    monkeypatch.setattr(chat_agent, "upload_file_to_gemini", fake_upload)
    part = await chat_agent.build_gemini_media_part(
        b"\x89PNG small", "image/png", "bench.png"
    )
    assert part["type"] == "image"
    assert part["data"]
    assert "uri" not in part


@pytest.mark.asyncio
async def test_build_gemini_media_part_uses_files_api_when_large(monkeypatch):
    async def fake_upload(**kwargs):
        assert kwargs["filename"] == "huge.png"
        return {
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc",
            "mime_type": "image/png",
            "name": "files/abc",
            "state": "ACTIVE",
        }

    monkeypatch.setattr(chat_agent, "_INLINE_CHAT_MEDIA_MAX_BYTES", 4)
    monkeypatch.setattr(chat_agent, "upload_file_to_gemini", fake_upload)
    part = await chat_agent.build_gemini_media_part(
        b"12345", "image/png", "huge.png"
    )
    assert part["uri"] == "https://generativelanguage.googleapis.com/files/abc"
    assert "data" not in part


def test_interactions_file_api_uri_normalizes_sdk_shapes():
    want = "https://generativelanguage.googleapis.com/files/abc"
    assert chat_agent.interactions_file_api_uri(
        "https://generativelanguage.googleapis.com/v1beta/files/abc",
        "files/abc",
    ) == want
    assert chat_agent.interactions_file_api_uri("", "files/abc") == want
    assert chat_agent.interactions_file_api_uri(want, "files/abc") == want
    assert chat_agent.interactions_file_api_uri("files/abc") == want


def test_apply_chat_media_resolution_skips_file_uri(monkeypatch):
    monkeypatch.setattr(chat_agent, "chat_media_resolution", lambda: "high")
    out = chat_agent.apply_chat_media_resolution([
        {"type": "text", "text": "hi"},
        {"type": "image", "uri": "https://x/files/a", "mime_type": "image/png"},
        {"type": "image", "data": "abc", "mime_type": "image/png"},
    ])
    assert "resolution" not in out[1]
    assert out[2]["resolution"] == "high"


@pytest.mark.asyncio
async def test_gemini_permission_error_is_user_visible(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setattr(llm_provider, "use_claude", lambda: False)
    monkeypatch.setattr(chat_agent, "get_chat_tools", lambda user_id=None: [])
    monkeypatch.setattr(chat_agent, "get_chat_system_instruction", lambda: "sys")
    monkeypatch.setattr(chat_agent, "cfg", lambda key, default=None: default)
    monkeypatch.setattr(
        chat_agent,
        "cfg_str",
        lambda key, default="": default if key != "CHAT_THINKING_LEVEL" else "",
    )
    monkeypatch.setattr(chat_agent, "log_error", AsyncMock())
    monkeypatch.setattr(chat_agent, "_clear_stored_interaction_id", AsyncMock())

    class _Err:
        event_type = "error"
        index = None
        interaction_id = "v1_failed"
        error = MagicMock(message="The caller does not have permission")
        interaction = MagicMock(id="v1_failed")

    async def fake_create(**kwargs):
        async def _stream():
            yield _Err()

        return _stream()

    mock_client = MagicMock()
    mock_client.aio.interactions.create = AsyncMock(side_effect=fake_create)
    monkeypatch.setattr(chat_agent, "get_client", lambda: mock_client)

    chunks = [
        chunk
        async for chunk in chat_agent._stream_chat_inner(
            [{"type": "text", "text": "see attached"}, {"type": "image", "uri": "x"}],
            allow_tools=False,
            user_id=10,
        )
    ]
    assert any("could not read the attached image" in c for c in chunks)
    assert not any('"type": "done"' in c for c in chunks)
