"""Client-side session/context management for the dev designer agent loop."""
from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Optional

from PIL import Image

from app.benchmark.session_config import DesignerSessionConfig

logger = logging.getLogger(__name__)

# Rough chars-per-token heuristic for budgeting (English + JSON).
_CHARS_PER_TOKEN = 4
_IMAGE_TOKEN_ESTIMATE = 800


def estimate_message_tokens(messages: list[dict[str, Any]], *, system: str = "") -> int:
    """Cheap token estimate for pre-call budgeting (not exact)."""
    chars = len(system or "")
    for msg in messages:
        chars += len(str(msg.get("text") or ""))
        result = msg.get("result")
        if isinstance(result, dict):
            chars += len(json.dumps(result, ensure_ascii=False))
        elif result is not None:
            chars += len(str(result))
        chars += len(msg.get("images") or []) * (_IMAGE_TOKEN_ESTIMATE * _CHARS_PER_TOKEN)
        state = msg.get("provider_state")
        if state is not None:
            chars += len(str(state)) // 2
    return max(1, chars // _CHARS_PER_TOKEN)


def compact_designer_tool_result(
    tool_name: str,
    result: Any,
    *,
    max_chars: int,
) -> Any:
    """Shrink large tool payloads before they enter model context."""
    if not isinstance(result, dict):
        raw = json.dumps(result, ensure_ascii=False) if result is not None else ""
        if len(raw) <= max_chars:
            return result
        return {"_truncated": True, "_original_chars": len(raw), "summary": raw[:max_chars]}

    if tool_name in {"ai_search", "keyword_search"}:
        hits = result.get("hits")
        if isinstance(hits, list):
            compact_hits = []
            for hit in hits[:8]:
                if not isinstance(hit, dict):
                    continue
                compact_hits.append({
                    "id": hit.get("id"),
                    "name": hit.get("name"),
                    "filename": hit.get("filename"),
                    "level": hit.get("level"),
                    "tags": (hit.get("tags") or [])[:12] if isinstance(hit.get("tags"), list) else hit.get("tags"),
                })
            return {
                **{k: v for k, v in result.items() if k != "hits"},
                "hits": compact_hits,
                "_compacted_for_model": True,
            }

    if tool_name == "get_image":
        images = result.get("images")
        if isinstance(images, list):
            return {
                **result,
                "images": [
                    {k: v for k, v in (img if isinstance(img, dict) else {}).items() if k != "url"}
                    for img in images[:8]
                ],
                "_compacted_for_model": True,
            }

    if tool_name in {"generate_multi_image", "generate_multi_image_pipeline"}:
        out = dict(result)
        out.pop("output_images", None)
        prompt = str(out.get("final_prompt") or out.get("prompt") or "")
        if len(prompt) > 2000:
            out["final_prompt"] = prompt[:2000] + "…"
        out["_compacted_for_model"] = True
        return out

    raw = json.dumps(result, ensure_ascii=False)
    if len(raw) <= max_chars:
        return result
    return {
        "_truncated": True,
        "_original_chars": len(raw),
        "summary": raw[:max_chars],
        "tool": tool_name,
    }


def strip_old_tool_images(
    messages: list[dict[str, Any]],
    *,
    retain_recent_tool_messages: int,
) -> list[dict[str, Any]]:
    """Drop PIL attachments from older tool turns; keep text/JSON only."""
    if retain_recent_tool_messages <= 0:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("images"):
                copy = dict(msg)
                copy.pop("images", None)
                copy["_images_stripped"] = True
                out.append(copy)
            else:
                out.append(msg)
        return out

    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool" and m.get("images")]
    keep = set(tool_indices[-retain_recent_tool_messages:])
    out = []
    for i, msg in enumerate(messages):
        if i in tool_indices and i not in keep:
            copy = dict(msg)
            copy.pop("images", None)
            copy["_images_stripped"] = True
            out.append(copy)
        else:
            out.append(msg)
    return out


def trim_message_history(messages: list[dict[str, Any]], *, max_messages: int) -> list[dict[str, Any]]:
    """Keep the opening user task + recent tail without splitting tool pairs."""
    if len(messages) <= max_messages:
        return messages
    head = messages[0:1]
    tail_budget = max(1, max_messages - 1)
    trimmed = messages[-tail_budget:]
    while trimmed:
        first = trimmed[0]
        if first.get("role") != "user":
            trimmed = trimmed[1:]
            continue
        if first.get("role") == "tool":
            trimmed = trimmed[1:]
            continue
        break
    return head + trimmed


def _middle_out_summary_block(messages: list[dict[str, Any]], summary: str) -> list[dict[str, Any]]:
    if not messages:
        return messages
    head = messages[0:1]
    tail = messages[-8:] if len(messages) > 9 else messages[1:]
    return head + [{
        "role": "user",
        "text": (
            "[Session summary — earlier turns compacted to stay within context budget]\n"
            + summary.strip()
        ),
    }] + tail


async def maybe_compact_messages(
    messages: list[dict[str, Any]],
    *,
    config: DesignerSessionConfig,
    system: str,
    router: str,
    summarize: Optional[Callable[[list[dict[str, Any]], str], Awaitable[str]]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Apply client-side compaction before a model turn.

    Returns (possibly new messages list, metadata about what ran).
    """
    meta: dict[str, Any] = {"actions": []}
    if not messages:
        return messages, meta

    policy = config.context_policy
    estimated = estimate_message_tokens(messages, system=system)
    meta["estimated_tokens"] = estimated
    trigger = int(config.context_token_budget * config.context_trigger_ratio)
    meta["trigger_tokens"] = trigger
    over_budget = estimated >= trigger

    working = list(messages)

    if policy in {"none", "provider_native"} and not over_budget:
        return working, meta

    if policy in {"auto", "compact_tools", "trim", "summarize"} or over_budget:
        working = strip_old_tool_images(
            working,
            retain_recent_tool_messages=config.retain_recent_image_turns,
        )
        meta["actions"].append("strip_old_images")

    if policy in {"auto", "compact_tools", "trim", "summarize"} or over_budget:
        compacted: list[dict[str, Any]] = []
        for msg in working:
            if msg.get("role") != "tool":
                compacted.append(msg)
                continue
            copy = dict(msg)
            copy["result"] = compact_designer_tool_result(
                str(msg.get("name") or "tool"),
                msg.get("result"),
                max_chars=config.max_tool_result_chars,
            )
            compacted.append(copy)
        working = compacted
        meta["actions"].append("compact_tool_results")

    if policy in {"auto", "trim", "compact_tools", "summarize"} or over_budget:
        before = len(working)
        working = trim_message_history(working, max_messages=config.max_messages)
        if len(working) < before:
            meta["actions"].append("trim_messages")
            meta["messages_dropped"] = before - len(working)

    should_summarize = (
        policy == "summarize"
        or (policy == "auto" and router == "gemini_native" and over_budget)
        or (over_budget and policy in {"compact_tools", "trim"})
    )
    instructions = config.effective_summarizer_instructions()
    if should_summarize and instructions and summarize is not None and len(working) > 10:
        try:
            summary = await summarize(working, instructions)
            if summary.strip():
                working = _middle_out_summary_block(working, summary)
                meta["actions"].append("llm_summarize")
                meta["estimated_tokens_after"] = estimate_message_tokens(working, system=system)
        except Exception as exc:
            logger.warning("[DesignerSession] summarization failed: %s", exc)
            meta["summarize_error"] = str(exc)

    return working, meta


def append_tool_message(
    messages: list[dict[str, Any]],
    *,
    tool_call_id: Any,
    name: str,
    result: Any,
    images: Optional[list[Image.Image]] = None,
    image_caption: Optional[str] = None,
    config: DesignerSessionConfig,
) -> None:
    """Append a compacted tool result to the in-memory session."""
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "result": compact_designer_tool_result(name, result, max_chars=config.max_tool_result_chars),
        "images": images or [],
        "image_caption": image_caption,
    })
