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
