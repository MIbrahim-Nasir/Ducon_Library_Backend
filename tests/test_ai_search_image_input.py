"""AISearch tool schemas accept text, image_ref, or both for multimodal search."""
from __future__ import annotations

from app.search_tools import (
    ai_search_interactions_tool,
    ai_search_live_declaration,
)


def test_chat_aisearch_schema_supports_image_ref():
    tool = ai_search_interactions_tool()
    props = tool["parameters"]["properties"]
    assert "query" in props
    assert "image_ref" in props
    assert "image" in props["image_ref"]["description"].lower() or "upload" in props["image_ref"]["description"].lower()
    # Neither is strictly required — text, image, or both.
    assert "query" not in tool["parameters"].get("required", [])
    assert "image_ref" not in tool["parameters"].get("required", [])
    desc = tool["description"].lower()
    assert "text" in desc and "image" in desc


def test_voice_aisearch_schema_supports_image_ref():
    tool = ai_search_live_declaration()
    props = tool["parameters"]["properties"]
    assert "query" in props
    assert "image_ref" in props
    desc = tool["description"].lower()
    assert "image_ref" in desc or "image" in desc


def test_search_endpoint_accepts_query_and_file_modes():
    """Documented contract: POST /search uses text / image / multimodal embeddings."""
    import inspect
    from app.main import search

    sig = inspect.signature(search)
    assert "query" in sig.parameters
    assert "file" in sig.parameters
