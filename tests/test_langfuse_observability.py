"""Langfuse observability — fail-open + dual-write from usage_recorder."""
from __future__ import annotations

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
    mock_obs.end.assert_called_once()


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
    mock_client.set_current_trace_io.assert_called()


def test_usage_recorder_dual_writes_when_enabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    recorded: list[dict] = []

    def _fake_record_generation(**kwargs):
        recorded.append(kwargs)

    monkeypatch.setattr(lf, "record_generation", _fake_record_generation)
    # Ensure get_client path is not needed
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


def test_smoke_import_main_with_langfuse_disabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    lf.reset_for_tests()
    # Import only — do not reload (main has heavy module-level side effects).
    import app.main as main_mod

    assert main_mod.app is not None
    assert lf.is_enabled() is False
