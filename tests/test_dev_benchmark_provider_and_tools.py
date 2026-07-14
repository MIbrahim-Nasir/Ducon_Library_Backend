import asyncio
import os

import pytest


def test_openrouter_payload_uses_reasoning_effort():
    from app.benchmark.provider_clients import build_openrouter_payload

    payload = build_openrouter_payload(
        model_id="deepseek/deepseek-r1",
        messages=[{"role": "user", "content": "hi"}],
        thinking="high",
        max_tokens=123,
        response_format={"type": "json_object"},
    )

    assert payload["model"] == "deepseek/deepseek-r1"
    assert payload["reasoning"] == {"effort": "high", "exclude": False}
    assert payload["max_tokens"] == 123
    assert payload["response_format"] == {"type": "json_object"}


def test_openrouter_payload_context_compression_plugin():
    from app.benchmark.provider_clients import build_openrouter_payload

    payload = build_openrouter_payload(
        model_id="test/model",
        messages=[{"role": "user", "content": "hi"}],
        context_compression=True,
    )
    assert payload["plugins"] == [{"id": "context-compression", "enabled": True}]


def test_parse_session_config_defaults():
    from app.benchmark.session_config import parse_session_config

    cfg = parse_session_config({})
    assert cfg.max_tokens == 8192
    assert cfg.context_policy == "auto"
    assert cfg.openrouter_context_compression is True
    assert cfg.claude_compaction_trigger_tokens == 150_000


def test_designer_config_includes_session_defaults():
    from app.benchmark.designer_agent import get_designer_config

    cfg = get_designer_config()
    assert "session" in cfg
    assert "auto" in cfg["session"]["context_policies"]
    assert "schema" in cfg["session"]
    assert "fields" in cfg["session"]["schema"]
    assert "applicability" in cfg["session"]["schema"]


def test_session_config_schema_endpoint_metadata():
    from app.benchmark.session_config import session_config_schema

    schema = session_config_schema()
    assert schema["routers"] == ["gemini_native", "claude_native", "openrouter"]
    assert "max_tokens" in schema["limits"]
    assert schema["limits"]["max_tokens"]["max"] == 128_000
    assert schema["limits"]["context_token_budget"]["max"] == 1_048_576
    assert schema["limits"]["claude_compaction_trigger_tokens"] == {
        "min": 50_000,
        "max": 1_000_000,
        "default": 150_000,
    }

    field_keys = {f["key"] for f in schema["fields"]}
    assert "openrouter_context_compression" in field_keys
    assert "claude_compaction" in field_keys
    assert schema["applicability"]["openrouter_context_compression"] == ["openrouter"]
    assert schema["applicability"]["claude_compaction"] == ["claude_native"]

    or_field = next(f for f in schema["fields"] if f["key"] == "openrouter_context_compression")
    assert or_field["type"] == "boolean"
    assert "description" in or_field and len(or_field["description"]) > 20
    assert or_field["routers"] == ["openrouter"]
    claude_field = next(f for f in schema["fields"] if f["key"] == "claude_compaction")
    assert claude_field["routers"] == ["claude_native"]


def test_claude_chat_with_tools_uses_beta_for_compaction(monkeypatch):
    from app.benchmark.provider_clients import _claude_chat_with_tools
    from app.benchmark.provider_registry import make_pair
    from app.benchmark.session_config import DesignerSessionConfig

    calls: list[tuple[str, dict]] = []

    class _StreamCtx:
        def __init__(self, msg):
            self._msg = msg

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get_final_message(self):
            return self._msg

    class _BetaMessages:
        def stream(self, **kwargs):
            calls.append(("beta", kwargs))
            raise RuntimeError("beta unavailable in test")

    class _Messages:
        def stream(self, **kwargs):
            calls.append(("stable", kwargs))

            class _Msg:
                content = []
                stop_reason = "end_turn"

            return _StreamCtx(_Msg())

    class _Client:
        beta = type("Beta", (), {"messages": _BetaMessages()})()
        messages = _Messages()

    monkeypatch.setattr(
        "app.benchmark.provider_clients.llm_provider.get_async_anthropic_client",
        lambda: _Client(),
    )
    monkeypatch.setattr(
        "app.benchmark.provider_clients.llm_provider.tool_use_blocks",
        lambda _msg: [],
    )
    monkeypatch.setattr(
        "app.benchmark.provider_clients.llm_provider.extract_text",
        lambda _msg: "",
    )
    monkeypatch.setattr(
        "app.benchmark.provider_clients.llm_provider.serialize_content",
        lambda _msg: [],
    )

    pair = make_pair(router="claude_native", model_id="claude-sonnet-4-6")
    session_cfg = DesignerSessionConfig(claude_compaction="enabled")

    asyncio.run(
        _claude_chat_with_tools(
            pair=pair,
            system="sys",
            messages=[{"role": "user", "text": "hi"}],
            tools=[],
            thinking=None,
            max_tokens=1024,
            session_config=session_cfg,
        )
    )

    assert len(calls) == 2
    assert calls[0][0] == "beta"
    assert "context_management" in calls[0][1]
    assert calls[1][0] == "stable"
    assert "context_management" not in calls[1][1]


def test_maybe_compact_messages_trims_when_over_budget():
    from app.benchmark.session_config import DesignerSessionConfig
    from app.benchmark.session_manager import estimate_message_tokens, maybe_compact_messages

    messages = [{"role": "user", "text": "task"}]
    for i in range(80):
        messages.append({"role": "assistant", "text": f"turn {i} " + ("x" * 500)})
        messages.append({"role": "tool", "name": "ai_search", "result": {"hits": [{"id": i}]}})

    config = DesignerSessionConfig(
        max_messages=20,
        context_token_budget=1000,
        context_trigger_ratio=0.5,
        context_policy="compact_tools",
    )
    compacted, meta = asyncio.run(
        maybe_compact_messages(messages, config=config, system="sys", router="gemini_native")
    )
    assert len(compacted) < len(messages)
    assert "compact_tool_results" in meta.get("actions", [])
    assert estimate_message_tokens(compacted, system="sys") <= estimate_message_tokens(messages, system="sys")


    from app.benchmark.provider_clients import build_openrouter_payload

    payload = build_openrouter_payload(
        model_id="deepseek/deepseek-r1",
        messages=[{"role": "user", "content": "hi"}],
        thinking="high",
        max_tokens=123,
        response_format={"type": "json_object"},
    )

    assert payload["model"] == "deepseek/deepseek-r1"
    assert payload["reasoning"] == {"effort": "high", "exclude": False}
    assert payload["max_tokens"] == 123
    assert payload["response_format"] == {"type": "json_object"}


def test_provider_registry_saves_router_model_pair_together(tmp_path, monkeypatch):
    from app.benchmark import provider_registry

    monkeypatch.setattr(provider_registry, "_REGISTRY_FILE", tmp_path / "pairs.json")
    async def no_metadata(model_id):
        return None
    monkeypatch.setattr(provider_registry, "fetch_openrouter_model_metadata", no_metadata)

    pair = asyncio.run(
        provider_registry.save_model_pair(
            {"router": "openrouter", "model_id": "deepseek/deepseek-r1"}
        )
    )
    pairs = asyncio.run(provider_registry.list_model_pairs())

    assert pair.id == "openrouter:deepseek/deepseek-r1"
    assert any(p.id == pair.id and p.router == "openrouter" for p in pairs)
    assert "high" in pair.thinking_modes


def test_thinking_modes_are_conservative_for_unknown_openrouter_model():
    from app.benchmark.provider_registry import infer_thinking_modes

    modes, supports_reasoning = infer_thinking_modes("openrouter", "vendor/unknown-chat")

    assert modes == ["none"]
    assert supports_reasoning is False


def test_legacy_combo_normalization_keeps_existing_benchmark_flow():
    from app.benchmark import store

    combo = store._normalize_combo(
        {
            "name": "legacy",
            "flow": "agent_loop",
            "image_model": "gemini-3-pro-image-preview",
            "prompt_model": "gemini-3-flash-preview",
        }
    )

    assert combo["image_model_pair"]["router"] == "gemini_native"
    assert combo["image_model_pair"]["model_id"] == "gemini-3-pro-image-preview"
    assert combo["prompt_model_pair"]["model_id"] == "gemini-3-flash-preview"


def test_scoped_filesystem_tool_blocks_traversal_and_symlink_escape(tmp_path):
    from app.benchmark.filesystem_tool import FilesystemScopeError, ScopedFilesystemTool

    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    tool = ScopedFilesystemTool(root)

    tool.call("write_text", path="plans/plan.txt", content="ok")
    assert tool.call("read_text", path="plans/plan.txt")["text"] == "ok"

    with pytest.raises(FilesystemScopeError):
        tool.call("read_text", path="../outside/secret.txt")

    if hasattr(os, "symlink"):
        link = root / "link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            return
        with pytest.raises(FilesystemScopeError):
            tool.call("read_text", path="link/secret.txt")


def test_designer_default_tool_config_contains_expected_tools():
    from app.benchmark.designer_agent import (
        DEFAULT_DESIGNER_TOOL_ACCESS,
        DESIGNER_TOOL_CATALOG,
    )

    tools = set(DEFAULT_DESIGNER_TOOL_ACCESS)
    assert {"ai_search", "keyword_search", "generate_multi_image", "filesystem"}.issubset(tools)
    assert "generate_multi_image_pipeline" in DESIGNER_TOOL_CATALOG
    assert "generate_multi_image_pipeline" not in DEFAULT_DESIGNER_TOOL_ACCESS


def test_designer_config_exposes_tools_and_system_prompt_sections():
    from app.benchmark.designer_agent import get_designer_config

    config = get_designer_config()
    assert "ai_search" in config["tools"]
    assert "generate_multi_image_pipeline" in config["tools"]
    assert "generate_multi_image" in config["default_tool_access"]
    assert "generate_multi_image_pipeline" not in config["default_tool_access"]
    assert len(config["system_prompt"]["sections"]) >= 4
    assert config["system_prompt"]["composed_default"]
    assert "list" in config["filesystem"]["operations"]


def test_system_prompt_replace_mode_uses_override_only():
    from app.benchmark.designer_agent import _system_prompt

    custom = "Only this prompt should be sent."
    assert _system_prompt(custom, mode="replace") == custom


def test_system_prompt_compose_mode_applies_section_overrides():
    from app.benchmark.designer_agent import _system_prompt

    composed = _system_prompt(
        None,
        mode="compose",
        section_overrides={"header": "Custom header for dev run."},
    )
    assert "Custom header for dev run." in composed


def test_designer_default_tool_config_contains_only_multi_image_generation():
    from app.benchmark.designer_agent import DEFAULT_DESIGNER_TOOLS

    tools = set(DEFAULT_DESIGNER_TOOLS)
    assert "generate_multi_image" in tools
    # The dev designer uses the multi-image pipeline only — it must NOT expose
    # the legacy single-image or two-image combine tools.
    assert "generate_image" not in tools
    assert "combine_images" not in tools


def test_designer_config_exposes_max_generation_rounds():
    from app.benchmark.designer_agent import get_designer_config

    config = get_designer_config()
    gen = config.get("generation") or {}
    assert int(gen.get("default_max_generation_rounds") or 0) >= 1


def test_claude_effective_max_tokens_respects_thinking_budget():
    from app.benchmark.provider_clients import (
        _CLAUDE_THINKING_BUDGET,
        _claude_effective_max_tokens,
        _claude_thinking_config,
    )

    thinking = _claude_thinking_config("high")
    assert thinking is not None
    effective = _claude_effective_max_tokens(thinking, 2000)
    assert effective > _CLAUDE_THINKING_BUDGET


def test_claude_messages_batch_tool_results():
    from app.benchmark.provider_clients import _claude_messages_with_tools

    messages = [
        {"role": "assistant", "text": None, "tool_calls": [
            {"id": "a", "name": "ai_search", "args": {"query": "pavers"}},
            {"id": "b", "name": "get_image", "args": {"image_ids": [1]}},
        ]},
        {"role": "tool", "tool_call_id": "a", "name": "ai_search", "result": {"hits": []}},
        {"role": "tool", "tool_call_id": "b", "name": "get_image", "result": {"images": []}, "images": []},
    ]
    claude_msgs = _claude_messages_with_tools(messages)
    assert len(claude_msgs) == 2
    tool_user = claude_msgs[1]
    assert tool_user["role"] == "user"
    assert len(tool_user["content"]) == 2
    assert tool_user["content"][0]["type"] == "tool_result"
    assert tool_user["content"][1]["type"] == "tool_result"


def test_designer_unlimited_budget_sentinels():
    from app.benchmark.designer_agent import (
        DEFAULT_MAX_TURNS,
        WALL_CLOCK_BUDGET_S,
        _resolve_max_turns,
        _resolve_wall_clock_budget_s,
    )

    assert _resolve_max_turns(None) == DEFAULT_MAX_TURNS
    assert _resolve_max_turns(0) is None
    assert _resolve_max_turns(10) == 10
    assert _resolve_wall_clock_budget_s(None) == WALL_CLOCK_BUDGET_S
    assert _resolve_wall_clock_budget_s(0) is None
    assert _resolve_wall_clock_budget_s(120) == 120


def test_reveal_endpoint_rejects_path_traversal():
    from app.routers.dev_benchmark import _resolve_dev_url

    assert _resolve_dev_url("/dev/outputs/../../etc/passwd") is None
    assert _resolve_dev_url("/dev/uploads/../../etc/passwd") is None
    # Unrecognized prefix → None.
    assert _resolve_dev_url("/public/images/foo.png") is None


def test_reveal_endpoint_resolves_outputs_url(tmp_path, monkeypatch):
    from app.benchmark import store
    # Redirect store dirs to tmp_path so the test never touches real benchmark_data.
    data_dir = tmp_path / "benchmark_data"
    outputs_dir = data_dir / "outputs"
    uploads_dir = data_dir / "uploads"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(store, "DATA_DIR", data_dir)
    monkeypatch.setattr(store, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(store, "UPLOADS_DIR", uploads_dir)

    from app.routers.dev_benchmark import _resolve_dev_url

    run_id = "run-rg"
    proc_id = "proc-p"
    target = outputs_dir / run_id / proc_id / "0.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\x89PNG\r\n\x1a\n")

    resolved = _resolve_dev_url(f"/dev/outputs/{run_id}/{proc_id}/0.png")
    assert resolved is not None
    assert resolved == target.resolve()
    assert resolved.is_relative_to(data_dir.resolve())

    up_target = uploads_dir / "abc.png"
    up_target.write_bytes(b"x")
    resolved_up = _resolve_dev_url("/dev/uploads/abc.png")
    assert resolved_up is not None
    assert resolved_up == up_target.resolve()


def test_run_dev_designer_job_agentic_loop(tmp_path, monkeypatch):
    """The agent loop executes tool calls the model asks for and stops on finish."""
    import io as _io
    from PIL import Image
    from app.benchmark import designer_agent, store
    from app.benchmark.provider_registry import make_pair

    # Keep all benchmark_data writes inside tmp_path.
    data_dir = tmp_path / "benchmark_data"
    monkeypatch.setattr(store, "DATA_DIR", data_dir)
    monkeypatch.setattr(store, "RUNS_DIR", data_dir / "runs")
    monkeypatch.setattr(store, "DESIGNER_JOBS_DIR", data_dir / "designer_jobs")
    monkeypatch.setattr(store, "UPLOADS_DIR", data_dir / "uploads")
    monkeypatch.setattr(store, "OUTPUTS_DIR", data_dir / "outputs")

    # Small PNG upload so normalize_user_image() can decode it.
    buf = _io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, format="PNG")
    user_image_bytes = buf.getvalue()

    fs_root = tmp_path / "scratch"

    # Scripted model turns: think + write notes → generate → refine → finish.
    turns = [
        {
            "text": "Planning the design.",
            "tool_calls": [
                {"id": "t1", "name": "fs_write_text",
                 "args": {"path": "notes.md", "content": "# plan\nmodern patio"}},
            ],
        },
        {
            "text": "Generating first candidate.",
            "tool_calls": [
                {"id": "t2", "name": "generate_multi_image",
                 "args": {"prompt": "a ducon patio", "image_refs": ["user_photo"]}},
            ],
        },
        {
            "text": "Refining the candidate.",
            "tool_calls": [
                {"id": "t3", "name": "generate_multi_image",
                 "args": {"prompt": "fix paving", "image_refs": ["user_photo", "generation:1"]}},
            ],
        },
        {
            "text": "Done.",
            "tool_calls": [
                {"id": "t4", "name": "finish",
                 "args": {"summary": "Two candidates, second selected.", "selected_generation": 2}},
            ],
        },
    ]

    async def _fake_chat_with_tools(**kwargs):
        return turns.pop(0)

    monkeypatch.setattr(designer_agent.provider_clients, "chat_with_tools", _fake_chat_with_tools)

    async def _fake_generate_multi_image(**kwargs):
        return {"pil_images": [Image.new("RGB", (8, 8))]}

    monkeypatch.setattr(designer_agent, "generate_multi_image", _fake_generate_multi_image)

    # Avoid needing a real DB session around generate_multi_image.
    class _FakeSession:
        async def __aenter__(self):
            return None
        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(designer_agent, "async_session_maker", lambda: _FakeSession())

    model_pair = make_pair(router="gemini_native", model_id="gemini-3-flash-preview")
    image_pair = make_pair(router="gemini_native", model_id="gemini-3-pro-image-preview")

    job = designer_agent.create_job(input_meta={"prompt": "test"})
    job.persist = False
    try:
        asyncio.run(
            designer_agent.run_dev_designer_job(
                job=job,
                user_image_bytes=user_image_bytes,
                user_prompt="test prompt",
                model_pair=model_pair,
                image_pair=image_pair,
                thinking=None,
                image_thinking=None,
                system_prompt=None,
                system_prompt_mode="append",
                tool_access={"generate_multi_image", "filesystem"},
                filesystem_root=str(fs_root),
                aspect_ratio=None,
                max_generation_rounds=5,
                max_turns=10,
                persist=False,
            )
        )
        assert job.status == "completed"
        assert job.error is None
        assert job.final is not None

        # The agent actually wrote to its per-job scratch filesystem.
        job_scratch = fs_root / job.id
        assert (job_scratch / "notes.md").read_text(encoding="utf-8").startswith("# plan")

        gen = job.final.get("design_generation") or {}
        assert gen.get("rounds_run") == 2
        assert gen.get("selected_generation") == 2
        meta = job.final.get("metadata") or {}
        assert meta.get("stop_reason") == "agent_called_finish"
        assert meta.get("loop") == "agentic"

        event_types = [ev.get("type") for ev in job.events]
        assert "assistant_message" in event_types
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "final" in event_types
    finally:
        designer_agent.JOBS.pop(job.id, None)


def test_resolve_aspect_ratio_auto_and_fixed():
    from PIL import Image

    from app.tool_generate_image import resolve_aspect_ratio

    assert resolve_aspect_ratio("auto") == "auto"
    assert resolve_aspect_ratio("AUTO") == "auto"
    assert resolve_aspect_ratio("16:9") == "16:9"
    assert resolve_aspect_ratio(None) is None

    landscape = Image.new("RGB", (1600, 900))
    assert resolve_aspect_ratio("auto", reference_image=landscape) == "auto"
    assert resolve_aspect_ratio("", reference_image=landscape) == "auto"
    assert resolve_aspect_ratio("4:3 landscape", reference_image=landscape) == "4:3"
    assert resolve_aspect_ratio("weird", reference_image=landscape) == "16:9"


def test_designer_config_default_aspect_ratio_is_auto():
    from app.benchmark import designer_agent

    cfg = designer_agent.get_designer_config()
    assert cfg["generation"]["default_aspect_ratio"] == "auto"
    assert "auto" in cfg["generation"]["aspect_ratio_options"]


def test_pipeline_generation_tool_uses_full_verify_loop(monkeypatch):
    from PIL import Image

    from app.benchmark import designer_agent
    from app.benchmark.provider_registry import make_pair

    captured: dict = {}

    async def _fake_generate_multi_image(**kwargs):
        captured.update(kwargs)
        return {
            "pil_images": [Image.new("RGB", (8, 8))],
            "final_prompt": "expanded prompt",
            "approved": True,
            "retries": 1,
            "steps": [{"kind": "prompt_initial"}],
            "model_used": "gemini-3-pro-image-preview",
        }

    monkeypatch.setattr(designer_agent, "generate_multi_image", _fake_generate_multi_image)

    class _FakeSession:
        async def __aenter__(self):
            return None
        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(designer_agent, "async_session_maker", lambda: _FakeSession())

    job = designer_agent.create_job(input_meta={"prompt": "test"})
    toolbox = designer_agent._AgentToolbox(
        job=job,
        user_image=Image.new("RGB", (64, 48)),
        tool_access={"generate_multi_image_pipeline"},
        filesystem_root=None,
        model_pair=make_pair(router="gemini_native", model_id="gemini-3-flash-preview"),
        image_pair=make_pair(router="gemini_native", model_id="gemini-3-pro-image-preview"),
        thinking=None,
        image_thinking=None,
        aspect_ratio="auto",
        out_rg="test-rg",
        out_pid="designer",
        max_generations=3,
    )

    result, images, _caption = asyncio.run(
        toolbox._run_generation(
            tool_name="generate_multi_image_pipeline",
            args={"prompt": "modern patio", "image_refs": ["user_photo"]},
            use_pipeline=True,
        )
    )
    assert captured.get("enable_verify") is True
    assert captured.get("on_step") is not None
    assert result.get("mode") == "pipeline"
    assert result.get("approved") is True
    assert len(images) == 1


def test_resolve_job_filesystem_root_creates_per_job_subdir(tmp_path):
    from app.benchmark.designer_agent import resolve_job_filesystem_root

    base = tmp_path / "scratch"
    job_id = "abc-123"
    resolved = resolve_job_filesystem_root(str(base), job_id)
    assert resolved == str((base / job_id).resolve())
    assert (base / job_id).is_dir()


def test_cancel_job_marks_running_job_cancelled(tmp_path, monkeypatch):
    from app.benchmark import designer_agent, store

    data_dir = tmp_path / "benchmark_data"
    monkeypatch.setattr(store, "DATA_DIR", data_dir)
    monkeypatch.setattr(store, "DESIGNER_JOBS_DIR", data_dir / "designer_jobs")

    job = designer_agent.create_job(input_meta={"prompt": "cancel me"})
    job.status = "running"

    async def _slow_loop(**kwargs):
        await asyncio.sleep(60)
        return {"text": "", "tool_calls": []}

    monkeypatch.setattr(designer_agent.provider_clients, "chat_with_tools", _slow_loop)

    async def _run_and_cancel():
        task = asyncio.create_task(
            designer_agent.run_dev_designer_job(
                job=job,
                user_image_bytes=b"not-a-real-image",
                user_prompt="cancel me",
                model_pair=designer_agent.provider_registry.make_pair(
                    router="gemini_native", model_id="gemini-3-flash-preview"
                ),
                image_pair=designer_agent.provider_registry.make_pair(
                    router="gemini_native", model_id="gemini-3-pro-image-preview"
                ),
                thinking=None,
                image_thinking=None,
                system_prompt=None,
                tool_access=set(),
                filesystem_root=None,
                aspect_ratio=None,
                persist=False,
            )
        )
        job.task = task
        await asyncio.sleep(0.05)
        ok = await designer_agent.cancel_job(job.id)
        assert ok is True
        await task

    # Patch image decode so run_dev_designer_job gets past input_image emit quickly.
    monkeypatch.setattr(
        designer_agent,
        "_image_from_bytes",
        lambda _data: __import__("PIL").Image.new("RGB", (4, 4)),
    )

    asyncio.run(_run_and_cancel())
    assert job.status == "cancelled"
    assert any(ev.get("type") == "cancelled" for ev in job.events)
