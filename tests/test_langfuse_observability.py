"""Langfuse observability — fail-open + dual-write from usage_recorder."""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.observability import langfuse_client as lf


@pytest.fixture(autouse=True)
def _reset_langfuse(monkeypatch):
    lf.reset_for_tests()
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.delenv("LANGFUSE_SERVICE_NAME", raising=False)
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    yield
    lf.reset_for_tests()


def test_disabled_by_default_is_noop():
    assert lf.is_enabled() is False
    assert lf.get_client() is None
    lf.record_generation(agent="chat", model="m", input_tokens=1, output_tokens=2)
    with lf.observe_generation("x", model="m") as obs:
        assert obs is not None
        obs.update(output="ok")
    with lf.start_trace("t") as span:
        span.update(output="ok")
    lf.flush()
    lf.shutdown()


def test_missing_keys_disables_even_when_flag_true(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    assert lf.is_enabled() is False
    assert lf.get_client() is None
    lf.record_generation(agent="chat", model="gemini-x", input_tokens=3)


def test_enabled_with_mock_client_records_generation(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    mock_obs = MagicMock()
    mock_client = MagicMock()
    mock_client.start_observation.return_value = mock_obs

    lf._client = mock_client
    lf._init_attempted = True

    assert lf.is_enabled() is True
    lf.record_generation(
        agent="image_gen",
        model="gemini-3-pro-image-preview",
        provider="gemini",
        input_tokens=10,
        output_tokens=20,
        user_id=7,
        guest_session_id="g-1",
        latency_ms=123,
        cost_usd=0.01,
    )

    mock_client.start_observation.assert_called_once()
    kwargs = mock_client.start_observation.call_args.kwargs
    assert kwargs["as_type"] == "generation"
    assert kwargs["model"] == "gemini-3-pro-image-preview"
    assert kwargs["usage_details"] == {"input": 10, "output": 20}
    assert kwargs["metadata"]["agent"] == "image_gen"
    assert kwargs["metadata"]["user_id"] == 7
    # Dual-write must never leave I/O blank (UI null/undefined).
    assert kwargs["input"] is not None
    assert kwargs["input"]["agent"] == "image_gen"
    assert kwargs["output"] is not None
    assert kwargs["output"]["status"] == "success"
    assert kwargs["output"]["input_tokens"] == 10
    mock_obs.end.assert_called_once()


def test_observe_generation_sets_input_and_accepts_output_update(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    mock_gen = MagicMock()
    mock_gen.__enter__ = MagicMock(return_value=mock_gen)
    mock_gen.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.start_as_current_observation.return_value = mock_gen
    mock_client.start_observation.return_value = MagicMock()

    lf._client = mock_client
    lf._init_attempted = True

    with lf.observe_generation(
        "designer-agent-turn",
        model="gemini-3.5-flash",
        input={"messages": [{"role": "user", "text": "design a patio"}]},
        tags=["designer"],
    ) as gen:
        assert gen is mock_gen
        gen.update(output=lf.generation_output_summary(text="ok", tool_calls=[{"name": "finish"}]))
        lf.record_generation(agent="designer", model="gemini-3.5-flash", input_tokens=1)

    kwargs = mock_client.start_as_current_observation.call_args.kwargs
    assert kwargs["as_type"] == "generation"
    assert kwargs["name"] == "designer-agent-turn"
    assert kwargs["input"]["messages"][0]["text"] == "design a patio"
    gen.update.assert_called()
    # Dual-write suppressed while observe_generation is active.
    mock_client.start_observation.assert_not_called()


def test_observe_generation_suppresses_dual_write(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    mock_gen = MagicMock()
    mock_gen.__enter__ = MagicMock(return_value=mock_gen)
    mock_gen.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.start_as_current_observation.return_value = mock_gen
    mock_client.start_observation.return_value = MagicMock()

    lf._client = mock_client
    lf._init_attempted = True

    with lf.observe_generation("generate-response", model="claude-sonnet-4-6") as gen:
        assert gen is mock_gen
        lf.record_generation(agent="chat", model="claude-sonnet-4-6", input_tokens=1)

    mock_client.start_observation.assert_not_called()


def test_preview_user_text_strips_media():
    assert lf.preview_user_text([{"type": "text", "text": "hello"}, {"type": "image", "uri": "x"}]) == "hello"
    assert lf.preview_user_text([]) is None


def test_preview_agent_messages_omits_image_bytes():
    preview = lf.preview_agent_messages(
        [
            {"role": "user", "text": "brief", "images": [object()]},
            {"role": "assistant", "text": "plan", "tool_calls": [{"name": "ai_search", "args": {"q": "sofa"}}]},
        ]
    )
    assert preview["message_count"] == 2
    assert preview["messages"][0]["image_count"] == 1
    assert "images" not in preview["messages"][0]
    assert preview["messages"][1]["tool_calls"][0]["name"] == "ai_search"


def test_start_trace_sets_session_and_tags(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.start_as_current_observation.return_value = mock_span

    lf._client = mock_client
    lf._init_attempted = True

    with lf.start_trace(
        "chat-response",
        user_id=9,
        session_id="sess-1",
        tags=["chat"],
        input="hi there",
    ) as span:
        assert span is mock_span

    kwargs = mock_client.start_as_current_observation.call_args.kwargs
    assert kwargs["name"] == "chat-response"
    assert kwargs["input"] == "hi there"
    mock_span.update.assert_called()


def test_ensure_service_name_defaults(monkeypatch):
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.delenv("LANGFUSE_SERVICE_NAME", raising=False)
    name = lf._ensure_service_name()
    assert name == "ducon-library-backend"
    assert os.environ.get("OTEL_SERVICE_NAME") == "ducon-library-backend"


def test_usage_recorder_dual_writes_when_enabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    recorded: list[dict] = []

    def _fake_record_generation(**kwargs):
        recorded.append(kwargs)

    monkeypatch.setattr(lf, "record_generation", _fake_record_generation)
    from app.admin import usage_recorder as ur

    ur.record(
        agent="designer",
        model="gemini-3.5-flash",
        provider="gemini",
        input_tokens=5,
        output_tokens=9,
        user_id=42,
    )
    assert len(recorded) == 1
    assert recorded[0]["agent"] == "designer"
    assert recorded[0]["input_tokens"] == 5
    assert recorded[0]["user_id"] == 42


def test_designer_llm_turn_creates_observation(monkeypatch):
    """designer chat_with_tools (Gemini) opens designer-agent-turn with I/O."""
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    mock_gen = MagicMock()
    mock_gen.__enter__ = MagicMock(return_value=mock_gen)
    mock_gen.__exit__ = MagicMock(return_value=False)
    mock_client = MagicMock()
    mock_client.start_as_current_observation.return_value = mock_gen
    lf._client = mock_client
    lf._init_attempted = True

    part = SimpleNamespace(text="hello", thought=False, function_call=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content, finish_reason=SimpleNamespace(name="STOP"))
    response = SimpleNamespace(candidates=[candidate], usage_metadata=None)

    async def _gen(**kwargs):
        return response

    gemini_client = MagicMock()
    gemini_client.aio.models.generate_content = _gen

    monkeypatch.setattr("app.designer.llm.get_gemini_client", lambda: gemini_client)
    monkeypatch.setattr("app.designer.llm.llm_provider.use_claude", lambda: False)

    from app.designer.llm import chat_with_tools

    result = asyncio.run(
        chat_with_tools(
            system="sys",
            messages=[{"role": "user", "text": "brief"}],
            tools=[{"name": "finish", "description": "done", "parameters": {}}],
            model="gemini-test",
            user_id=1,
        )
    )
    assert result["text"] == "hello"
    kwargs = mock_client.start_as_current_observation.call_args.kwargs
    assert kwargs["name"] == "designer-agent-turn"
    assert kwargs["as_type"] == "generation"
    assert kwargs["input"] is not None
    assert "messages" in kwargs["input"]
    mock_gen.update.assert_called()
    out_call = [c for c in mock_gen.update.call_args_list if c.kwargs.get("output")]
    assert out_call
    assert out_call[0].kwargs["output"]["text"] == "hello"


def test_smoke_import_main_with_langfuse_disabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    lf.reset_for_tests()
    import app.main as main_mod

    assert main_mod.app is not None
    assert lf.is_enabled() is False
