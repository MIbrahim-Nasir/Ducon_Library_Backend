"""Unit tests for the autonomous designer agent (no live Gemini)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from PIL import Image


def _run(coro):
    return asyncio.run(coro)


def test_tool_declarations_register_expected_tools():
    from app.designer.tools import TOOL_NAMES, tool_declarations

    names = [t["name"] for t in tool_declarations()]
    assert names == list(TOOL_NAMES)
    for t in tool_declarations():
        assert "parameters" in t and t["parameters"].get("type") == "object"
        assert t.get("description")


def test_budget_snapshot_and_env_overrides(monkeypatch):
    from app.designer import budgets

    monkeypatch.setenv("DESIGNER_AGENT_SEARCH_LIMIT", "3")
    monkeypatch.setattr(budgets, "cfg", lambda key, default=None: default)

    snap = budgets.budget_snapshot()
    assert snap["max_turns"] == budgets.DEFAULT_MAX_TURNS
    assert snap["max_generations"] == budgets.DEFAULT_MAX_GENERATIONS
    assert snap["max_searches"] == budgets.DEFAULT_MAX_SEARCHES
    assert snap["wall_time_seconds"] == budgets.DEFAULT_WALL_TIME_S
    assert budgets.search_limit() == 3
    assert budgets.DEFAULT_SEARCH_LIMIT == 12
    assert budgets.DEFAULT_MAX_REFERENCES == 7
    assert budgets.SEARCH_LIMIT_HARD_MAX == 15
    assert budgets.max_references() == 7


def test_search_limit_hard_max(monkeypatch):
    from app.designer import budgets

    monkeypatch.setenv("DESIGNER_AGENT_SEARCH_LIMIT", "99")
    monkeypatch.setattr(budgets, "cfg", lambda key, default=None: default)
    assert budgets.search_limit() == budgets.SEARCH_LIMIT_HARD_MAX


def test_submit_plan_can_be_refined(monkeypatch):
    from app.designer.jobs import DesignerJob
    from app.designer.tools import DesignerToolbox

    job = DesignerJob(id="job-plan", user_id=1)
    plans: list[dict] = []

    async def fake_emit(j, event_type, **data):
        if event_type == "plan":
            plans.append(data.get("plan") or {})

    monkeypatch.setattr("app.designer.tools.emit", fake_emit)

    tb = DesignerToolbox(
        job=job,
        db=SimpleNamespace(),
        user_image=Image.new("RGB", (8, 8), "white"),
        embedding_model=object(),
        collection=object(),
        image_model="flash",
        aspect_ratio="16:9",
    )
    _run(
        tb.execute(
            "submit_plan",
            {"space_analysis": "terrace", "design_direction": "v1"},
        )
    )
    _run(
        tb.execute(
            "submit_plan",
            {"space_analysis": "terrace + pool", "design_direction": "v2 refined"},
        )
    )
    assert tb.plan["design_direction"] == "v2 refined"
    assert len(plans) == 2


def test_ai_search_passes_rich_limit(monkeypatch):
    from app.designer.jobs import DesignerJob
    from app.designer.tools import DesignerToolbox

    job = DesignerJob(id="job-search-limit", user_id=1)
    seen: dict[str, Any] = {}

    async def fake_emit(*_a, **_k):
        return None

    async def fake_ai_search(**kwargs):
        seen["limit"] = kwargs.get("limit")
        return [{"id": i, "name": f"h{i}", "label": f"h{i}"} for i in range(1, 13)]

    monkeypatch.setattr("app.designer.tools.emit", fake_emit)
    monkeypatch.setattr("app.designer.tools.ai_search", fake_ai_search)

    tb = DesignerToolbox(
        job=job,
        db=SimpleNamespace(),
        user_image=Image.new("RGB", (8, 8), "white"),
        embedding_model=object(),
        collection=object(),
        image_model="flash",
        aspect_ratio="16:9",
    )
    tb.search_limit = 12
    result, _, _ = _run(tb.execute("ai_search", {"query": "pergola", "limit": 20}))
    assert seen["limit"] == 15  # hard max
    assert len(result["hits"]) == 12


def test_max_references_safety_cap_keeps_model_order(monkeypatch):
    """Cap trims excess catalog refs the model asked for; does not force first-3 shortlist."""
    from app.designer.jobs import DesignerJob
    from app.designer.tools import DesignerToolbox

    job = DesignerJob(id="job-refs", user_id=1)

    async def fake_emit(*_a, **_k):
        return None

    monkeypatch.setattr("app.designer.tools.emit", fake_emit)

    tb = DesignerToolbox(
        job=job,
        db=SimpleNamespace(),
        user_image=Image.new("RGB", (8, 8), "white"),
        embedding_model=object(),
        collection=object(),
        image_model="flash",
        aspect_ratio="16:9",
    )
    tb.max_references = 7
    for i in range(1, 10):
        tb.sources[i] = {"id": i, "name": f"c{i}", "label": f"c{i}"}

    refs = ["user_photo"] + [f"catalog:{i}" for i in range(1, 10)]
    descriptors, chosen, errors = tb._resolve_refs(refs)
    assert not errors
    catalog = [d for d in descriptors if d.type == "catalog_id"]
    assert len(catalog) == 7
    assert [int(d.source) for d in catalog] == list(range(1, 8))
    assert len(chosen) == 7


def test_agent_loop_status_message_and_early_finish(monkeypatch):
    """Finish on turn 2 — does not pad to max turns; status omits max unless debug."""
    from app.designer.jobs import DesignerJob
    from app.designer import loop as loop_mod

    job = DesignerJob(id="job-early", user_id=3)
    events: list[dict[str, Any]] = []

    async def fake_emit(j, event_type, **data):
        events.append({"type": event_type, **data})

    monkeypatch.setattr(loop_mod, "emit", fake_emit)
    monkeypatch.setattr(loop_mod.budgets, "max_turns", lambda: 12)
    monkeypatch.setattr(loop_mod.budgets, "wall_time_seconds", lambda: 0)
    monkeypatch.setattr(loop_mod, "build_system_prompt", lambda: "sys")
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setattr(loop_mod, "_status_shows_max_turns", lambda: False)

    calls = {"n": 0}

    async def fake_chat_with_tools(**_k):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "text": "ok",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "submit_plan",
                        "args": {
                            "space_analysis": "ok",
                            "design_direction": "simple",
                        },
                    }
                ],
                "provider_state": None,
            }
        return {
            "text": "done",
            "tool_calls": [
                {
                    "id": "c2",
                    "name": "finish",
                    "args": {"summary": "Stopped early", "selected_generation": None},
                }
            ],
            "provider_state": None,
        }

    monkeypatch.setattr(loop_mod, "chat_with_tools", fake_chat_with_tools)

    async def fake_tool_emit(*_a, **_k):
        return None

    monkeypatch.setattr("app.designer.tools.emit", fake_tool_emit)

    toolbox, stop_reason, turns_used = _run(
        loop_mod.run_agent_loop(
            job=job,
            db=SimpleNamespace(),
            user_image=Image.new("RGB", (16, 16), "blue"),
            user_prompt="quick",
            embedding_model=object(),
            collection=object(),
            image_model="flash",
            aspect_ratio="16:9",
        )
    )

    assert stop_reason == "agent_called_finish"
    assert turns_used == 2
    assert calls["n"] == 2
    status_msgs = [e["message"] for e in events if e.get("type") == "status" and e.get("message")]
    assert status_msgs
    assert all("max" not in m.lower() for m in status_msgs)
    assert all("/12" not in m for m in status_msgs)
    assert any("Designer agent working (turn 1)" in m for m in status_msgs)
    assert any(e.get("max_turns") == 12 for e in events if e.get("type") == "status")
    assert toolbox.finish_args["summary"] == "Stopped early"


def test_turn_status_message_debug_includes_max(monkeypatch):
    from app.designer import loop as loop_mod

    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert "max 12" in loop_mod.turn_status_message(3, 12)
    monkeypatch.setenv("LOG_LEVEL", "info")
    monkeypatch.setattr(loop_mod, "_status_shows_max_turns", lambda: False)
    assert loop_mod.turn_status_message(3, 12) == "Designer agent working (turn 3)."


def test_keyword_search_declaration_encourages_dual_use():
    from app.designer.tools import tool_declarations

    by_name = {t["name"]: t for t in tool_declarations()}
    kw = by_name["keyword_search"]["description"].lower()
    ai = by_name["ai_search"]["description"].lower()
    assert "keyword_search" in ai or "also call keyword_search" in ai
    assert "exact" in kw or "modular" in kw
    assert "level" in kw and ("products" in kw or "category" in kw)
    props = by_name["keyword_search"]["parameters"]["properties"]
    assert "class" in props
    assert "category" in props
    assert props["level"].get("enum") == ["All", "Designs", "Products", "Areas"]


def test_system_prompt_includes_catalog_orientation(monkeypatch):
    from app.designer import prompts as prompts_mod
    from app import prompt_loader

    monkeypatch.setattr(prompt_loader, "DESIGNER_AGENT_SYSTEM", "You are the designer agent.")
    monkeypatch.setattr(prompt_loader, "DESIGNER_PROMPT_WRITER_SYSTEM", "")
    monkeypatch.setattr(prompt_loader, "ensure_prompts_loaded", lambda: None)
    monkeypatch.setattr(
        prompts_mod,
        "get_ducon_catalog_orientation",
        lambda: "## Ducon catalog orientation\nProduct category / type values: Pergola, Fountain.",
    )
    text = prompts_mod.build_system_prompt()
    assert "Ducon catalog orientation" in text
    assert "Pergola" in text
    assert "Soft harness budgets" in text


def test_ducon_catalog_orientation_uses_live_or_fallback(monkeypatch):
    from app import catalog_filter_context as cfc

    cfc.clear_catalog_context_cache()
    monkeypatch.setattr(cfc, "get_catalog_class_options", lambda: ("Kitchen",))
    monkeypatch.setattr(cfc, "get_catalog_type_options", lambda: ("Pergola",))
    monkeypatch.setattr(cfc, "get_catalog_theme_options", lambda: ("Modern Oasis",))
    monkeypatch.setattr(
        cfc, "get_catalog_feature_options", lambda: ("outdoor kitchen", "pergola lounge")
    )
    # Orientation is lru_cached; clear only that helper after patching getters.
    clearer = getattr(cfc.get_ducon_catalog_orientation, "cache_clear", None)
    if callable(clearer):
        clearer()
    text = cfc.get_ducon_catalog_orientation()
    assert "Ducon catalog orientation" in text
    assert "Pergola" in text
    assert "Kitchen" in text
    assert "outdoor kitchen" in text
    assert "live catalog metadata" in text

    monkeypatch.setattr(cfc, "get_catalog_class_options", lambda: ())
    monkeypatch.setattr(cfc, "get_catalog_type_options", lambda: ())
    monkeypatch.setattr(cfc, "get_catalog_theme_options", lambda: ())
    monkeypatch.setattr(cfc, "get_catalog_feature_options", lambda: ())
    clearer = getattr(cfc.get_ducon_catalog_orientation, "cache_clear", None)
    if callable(clearer):
        clearer()
    fallback = cfc.get_ducon_catalog_orientation()
    assert "curated Ducon summary" in fallback
    assert "Pergola" in fallback



def test_parse_selected_generation():
    from app.designer.tools import parse_selected_generation

    gens = [{"index": 1, "generation": {"id": 10}}, {"index": 2, "generation": {"id": 20}}]
    assert parse_selected_generation(2, gens)["generation"]["id"] == 20
    assert parse_selected_generation(99, gens)["generation"]["id"] == 20  # fallback last
    assert parse_selected_generation(None, gens)["generation"]["id"] == 20
    assert parse_selected_generation(1, []) is None


def test_search_budget_enforcement(monkeypatch):
    from app.designer.jobs import DesignerJob
    from app.designer.tools import DesignerToolbox

    job = DesignerJob(id="job1", user_id=1)
    emitted: list[str] = []

    async def fake_emit(j, event_type, **data):
        emitted.append(event_type)

    monkeypatch.setattr("app.designer.tools.emit", fake_emit)

    async def fake_ai_search(**_k):
        return [{"id": 1, "name": "hit", "label": "hit"}]

    monkeypatch.setattr("app.designer.tools.ai_search", fake_ai_search)

    tb = DesignerToolbox(
        job=job,
        db=SimpleNamespace(),
        user_image=Image.new("RGB", (8, 8), "white"),
        embedding_model=object(),
        collection=object(),
        image_model="flash",
        aspect_ratio="16:9",
    )
    tb.max_searches = 1
    tb.max_generations = 2

    r1, _, _ = _run(tb.execute("ai_search", {"query": "pergola"}))
    assert "hits" in r1
    assert r1["searches_remaining"] == 0

    r2, _, _ = _run(tb.execute("ai_search", {"query": "pool"}))
    assert r2.get("error") == "search budget exhausted"
    assert "search_started" in emitted
    assert "search_done" in emitted


def test_generation_budget_enforcement(monkeypatch):
    from app.designer.jobs import DesignerJob
    from app.designer.tools import DesignerToolbox

    job = DesignerJob(id="job2", user_id=1)

    async def fake_emit(*_a, **_k):
        return None

    monkeypatch.setattr("app.designer.tools.emit", fake_emit)

    tb = DesignerToolbox(
        job=job,
        db=SimpleNamespace(),
        user_image=Image.new("RGB", (8, 8), "white"),
        embedding_model=object(),
        collection=object(),
        image_model="flash",
        aspect_ratio="16:9",
    )
    tb.max_generations = 1
    tb.generations = [
        {
            "index": 1,
            "generation": {"id": 1},
            "pil": Image.new("RGB", (4, 4)),
            "prompt": "x",
            "references": [],
        }
    ]
    result, imgs, _ = _run(
        tb.execute(
            "generate_image",
            {"prompt": "design", "image_refs": ["user_photo"]},
        )
    )
    assert result.get("error") == "generation budget exhausted"
    assert imgs == []


def test_agent_loop_tool_calls_then_finish(monkeypatch):
    """Mock LLM: search → generate → finish; assert tools invoked and final fields."""
    from app.designer.jobs import DesignerJob
    from app.designer import loop as loop_mod

    job = DesignerJob(id="job-loop", user_id=7)
    events: list[dict[str, Any]] = []

    async def fake_emit(j, event_type, **data):
        events.append({"type": event_type, **data})

    monkeypatch.setattr(loop_mod, "emit", fake_emit)
    monkeypatch.setattr(loop_mod.budgets, "max_turns", lambda: 10)
    monkeypatch.setattr(loop_mod.budgets, "wall_time_seconds", lambda: 0)
    monkeypatch.setattr(loop_mod, "build_system_prompt", lambda: "sys")

    calls = {"n": 0}

    async def fake_chat_with_tools(**_k):
        calls["n"] += 1
        n = calls["n"]
        if n == 1:
            return {
                "text": "planning",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "submit_plan",
                        "args": {
                            "space_analysis": "villa terrace",
                            "design_direction": "modern pergola",
                            "generation_prompt_seed": "redesign with pergola",
                        },
                    },
                    {"id": "c2", "name": "ai_search", "args": {"query": "modern pergola terrace"}},
                ],
                "provider_state": None,
            }
        if n == 2:
            return {
                "text": "generating",
                "tool_calls": [
                    {
                        "id": "c3",
                        "name": "generate_image",
                        "args": {
                            "prompt": "Image 1 client space. Apply catalog:1 pergola.",
                            "image_refs": ["user_photo", "catalog:1"],
                        },
                    }
                ],
                "provider_state": None,
            }
        return {
            "text": "done",
            "tool_calls": [
                {
                    "id": "c4",
                    "name": "finish",
                    "args": {"summary": "Modern pergola redesign", "selected_generation": 1},
                }
            ],
            "provider_state": None,
        }

    monkeypatch.setattr(loop_mod, "chat_with_tools", fake_chat_with_tools)

    async def fake_ai_search(**_k):
        return [
            {
                "id": 1,
                "name": "Pergola A",
                "label": "Pergola A",
                "filename": "a.jpg",
                "url": "/a.jpg",
                "source": "ai_search",
                "query": "modern pergola terrace",
            }
        ]

    async def fake_generate_multi_image(**_k):
        return {
            "id": 99,
            "url": "gens/99.png",
            "signed_url": "https://example/99.png",
            "generation_name": "designer_job_test_1",
            "final_prompt": "final prompt",
        }

    async def fake_load(self, generation_id: int):
        return Image.new("RGB", (16, 16), "green")

    monkeypatch.setattr("app.designer.tools.ai_search", fake_ai_search)
    monkeypatch.setattr("app.designer.tools.generate_multi_image", fake_generate_multi_image)
    monkeypatch.setattr(
        "app.designer.tools.DesignerToolbox._load_generation_image",
        fake_load,
    )

    async def fake_tool_emit(*_a, **_k):
        return None

    monkeypatch.setattr("app.designer.tools.emit", fake_tool_emit)

    toolbox, stop_reason, turns_used = _run(
        loop_mod.run_agent_loop(
            job=job,
            db=SimpleNamespace(),
            user_image=Image.new("RGB", (32, 32), "blue"),
            user_prompt="Add a modern pergola",
            embedding_model=object(),
            collection=object(),
            image_model="flash",
            aspect_ratio="16:9",
        )
    )

    assert stop_reason == "agent_called_finish"
    assert turns_used >= 3
    assert toolbox.searches_used == 1
    assert len(toolbox.generations) == 1
    assert toolbox.generations[0]["generation"]["id"] == 99
    assert toolbox.finish_args["summary"] == "Modern pergola redesign"
    assert toolbox.plan is not None
    assert any(e["type"] == "tool_call" for e in events)
    assert any(e["type"] == "assistant_message" for e in events)


def test_final_image_fields_from_best_generation(monkeypatch):
    from app.designer.jobs import final_image_fields

    monkeypatch.setattr(
        "app.designer.jobs.storage.get_generation_url",
        lambda gid, url: f"https://signed/{gid}",
    )
    fields = final_image_fields(
        {"best_generation": {"id": 5, "url": "gens/5.png", "generation_name": "g"}}
    )
    assert fields["final_image_url"] == "https://signed/5"
    assert fields["final_image"]["id"] == 5
