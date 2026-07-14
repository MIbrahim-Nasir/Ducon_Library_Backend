"""
LLM provider abstraction — Gemini (default) vs Claude (Anthropic).

When the env flag ``USE_CLAUDE`` is truthy, the *text-reasoning* agents route
through Anthropic Claude instead of Gemini:

  • Chat agent
  • Image-generation prompt writer + verifier + QC evaluator (ImageGenAgent)
  • Designer job agent (analyze / evaluate / summarize + DesignerPromptSession)

Everything else stays on Gemini regardless of this flag:

  • AI search / embeddings
  • Voice agent (Gemini Live)
  • The actual image-generation models (Nano Banana Pro / Flash)
  • Studio directions agent (search-driven) and quotation analyzer

This module is the single place that knows about the Anthropic SDK. Agents call
the small helpers here so their own code stays provider-agnostic.

Model: Claude Sonnet 4.6 with adaptive ("high") extended thinking by default.
"""
from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Optional

from PIL import Image

try:  # The dependency is optional until USE_CLAUDE is enabled.
    import anthropic  # type: ignore
    _ANTHROPIC_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - import guard
    anthropic = None  # type: ignore
    _ANTHROPIC_IMPORT_ERROR = exc


from app.admin.settings_store import cfg, cfg_bool, cfg_str, cfg_int

# ── Config ────────────────────────────────────────────────────────────────────

def use_claude() -> bool:
    """True when text-reasoning agents should use Claude instead of Gemini."""
    return cfg_bool("USE_CLAUDE", False)


def claude_model() -> str:
    return cfg_str("CLAUDE_MODEL", "claude-sonnet-4-6")


def claude_max_tokens() -> int:
    return cfg_int("CLAUDE_MAX_TOKENS", 16000)


def _thinking_param() -> Optional[dict]:
    """
    Extended-thinking config. Defaults to adaptive ('high' reasoning on 4.6),
    which also enables interleaved thinking for tool use automatically.

    CLAUDE_THINKING overrides: 'adaptive' (default) | 'off' | '<int budget>'.
    """
    val = cfg_str("CLAUDE_THINKING", "adaptive").strip().lower()
    if val in ("off", "none", "0", "false", "no"):
        return None
    if val.isdigit():
        return {"type": "enabled", "budget_tokens": int(val)}
    return {"type": "adaptive"}


# ── Clients (lazy singletons) ──────────────────────────────────────────────────

_client = None
_aclient = None


def _require_sdk() -> None:
    if anthropic is None:
        raise RuntimeError(
            "USE_CLAUDE is enabled but the 'anthropic' package is not installed. "
            f"Run `pip install anthropic`. Import error: {_ANTHROPIC_IMPORT_ERROR}"
        )
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("USE_CLAUDE is enabled but ANTHROPIC_API_KEY is not set.")


def get_anthropic_client():
    global _client
    if _client is None:
        _require_sdk()
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def get_async_anthropic_client():
    global _aclient
    if _aclient is None:
        _require_sdk()
        _aclient = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _aclient


# ── Content-block builders ──────────────────────────────────────────────────────

def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def bytes_image_block(data: bytes, media_type: str = "image/jpeg") -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


def pil_image_block(img: Image.Image, *, quality: int = 92) -> dict:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return bytes_image_block(buf.getvalue(), "image/jpeg")


def to_claude_tools(gemini_tools: list[dict]) -> list[dict]:
    """Translate Interactions/Gemini flat tool dicts to Anthropic tool schema."""
    tools: list[dict] = []
    for t in gemini_tools or []:
        if t.get("type") not in (None, "function"):
            continue
        name = t.get("name")
        if not name:
            continue
        tools.append({
            "name": name,
            "description": t.get("description", "") or "",
            "input_schema": t.get("parameters") or {"type": "object", "properties": {}},
        })
    return tools


# ── Response helpers ─────────────────────────────────────────────────────────────

def extract_text(message: Any) -> str:
    """Concatenate all text blocks from a Claude Message (ignores thinking/tools)."""
    parts: list[str] = []
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts).strip()


def serialize_block(block: Any) -> dict:
    """
    Convert an SDK response content block into a plain dict that is valid as
    INPUT for the next request. Whitelists fields per type so that response-only
    fields (e.g. citations=null) never trip input validation, while preserving
    the thinking ``signature`` required for multi-turn thinking continuity.
    """
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": getattr(block, "text", "") or ""}
    if btype == "thinking":
        return {
            "type": "thinking",
            "thinking": getattr(block, "thinking", "") or "",
            "signature": getattr(block, "signature", "") or "",
        }
    if btype == "redacted_thinking":
        return {"type": "redacted_thinking", "data": getattr(block, "data", "")}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", None),
            "name": getattr(block, "name", None),
            "input": getattr(block, "input", {}) or {},
        }
    # Fallback — best effort.
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    return {"type": btype or "text", "text": str(block)}


def serialize_content(message: Any) -> list[dict]:
    return [serialize_block(b) for b in (getattr(message, "content", None) or [])]


def tool_use_blocks(message: Any) -> list[dict]:
    """Return the tool_use blocks of a message as {id, name, input} dicts."""
    out: list[dict] = []
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "tool_use":
            out.append({
                "id": getattr(block, "id", None),
                "name": getattr(block, "name", None),
                "input": getattr(block, "input", {}) or {},
            })
    return out


def parse_json_text(text: str | None) -> dict:
    """
    Tolerant JSON parse for Claude text output (mirrors gemini._parse_json_response):
    strips markdown fences, decodes the first JSON value, unwraps single-object
    arrays, and merges multi-object arrays.
    """
    if not text or not str(text).strip():
        raise ValueError("Empty JSON response from Claude")
    raw = str(text).strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    # Skip any prose before the first JSON object/array.
    start = raw.find("{")
    arr_start = raw.find("[")
    if arr_start != -1 and (start == -1 or arr_start < start):
        start = arr_start
    if start > 0:
        raw = raw[start:]

    try:
        obj, _end = decoder.raw_decode(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from Claude: {raw[:200]}") from exc

    if isinstance(obj, list):
        dict_items = [item for item in obj if isinstance(item, dict)]
        if len(dict_items) == 1:
            obj = dict_items[0]
        elif len(dict_items) > 1:
            merged: dict = {}
            for item in dict_items:
                merged.update(item)
            obj = merged
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object from Claude, got {type(obj).__name__}")
    return obj


# ── Core completion (non-streaming, via stream helper to avoid the long-request guard) ──

def _build_kwargs(
    *,
    system: str,
    messages: list[dict],
    tools: Optional[list[dict]],
    max_tokens: Optional[int],
    thinking: bool,
) -> dict:
    kwargs: dict[str, Any] = {
        "model": claude_model(),
        "max_tokens": max_tokens or claude_max_tokens(),
        "system": system or "",
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    th = _thinking_param() if thinking else None
    if th:
        kwargs["thinking"] = th
    return kwargs


def complete_message(
    *,
    system: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    max_tokens: Optional[int] = None,
    thinking: bool = True,
):
    """Synchronous full completion. Returns the final anthropic Message."""
    client = get_anthropic_client()
    kwargs = _build_kwargs(
        system=system, messages=messages, tools=tools,
        max_tokens=max_tokens, thinking=thinking,
    )
    with client.messages.stream(**kwargs) as stream:
        msg = stream.get_final_message()
    try:
        from app.admin.usage_helpers import record_from_usage_dict
        usage = getattr(msg, "usage", None)
        if usage is not None:
            u = usage.model_dump() if hasattr(usage, "model_dump") else usage
            record_from_usage_dict(u, agent="chat", model=claude_model(), provider="claude")
    except Exception:
        pass
    return msg


async def acomplete_message(
    *,
    system: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    max_tokens: Optional[int] = None,
    thinking: bool = True,
):
    """Async full completion. Returns the final anthropic Message."""
    client = get_async_anthropic_client()
    kwargs = _build_kwargs(
        system=system, messages=messages, tools=tools,
        max_tokens=max_tokens, thinking=thinking,
    )
    async with client.messages.stream(**kwargs) as stream:
        msg = await stream.get_final_message()
    try:
        from app.admin.usage_helpers import record_from_usage_dict
        usage = getattr(msg, "usage", None)
        if usage is not None:
            u = usage.model_dump() if hasattr(usage, "model_dump") else usage
            record_from_usage_dict(u, agent="chat", model=claude_model(), provider="claude")
    except Exception:
        pass
    return msg


# ── Convenience: single-turn text / JSON ─────────────────────────────────────────

def generate_text(system: str, content_blocks: list[dict], *, thinking: bool = True) -> str:
    msg = complete_message(
        system=system,
        messages=[{"role": "user", "content": content_blocks}],
        thinking=thinking,
    )
    return extract_text(msg)


async def agenerate_text(system: str, content_blocks: list[dict], *, thinking: bool = True) -> str:
    msg = await acomplete_message(
        system=system,
        messages=[{"role": "user", "content": content_blocks}],
        thinking=thinking,
    )
    return extract_text(msg)


def generate_json(system: str, content_blocks: list[dict], *, thinking: bool = True) -> dict:
    return parse_json_text(generate_text(system, content_blocks, thinking=thinking))


async def agenerate_json(system: str, content_blocks: list[dict], *, thinking: bool = True) -> dict:
    return parse_json_text(await agenerate_text(system, content_blocks, thinking=thinking))
