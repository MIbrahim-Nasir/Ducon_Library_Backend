"""Tool-calling LLM turns for the production designer agent (Gemini / Claude)."""
from __future__ import annotations

import io
import json
from typing import Any, Optional

from google.genai.types import (
    Content,
    FunctionDeclaration,
    GenerateContentConfig,
    Part,
    ThinkingConfig,
    Tool,
)
from PIL import Image

from app import llm_provider
from app.designer.budgets import designer_model
from app.gemini import get_gemini_client


def _pil_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _gemini_contents_from_messages(messages: list[dict[str, Any]]) -> list[Content]:
    contents: list[Content] = []
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            parts: list[Part] = []
            for img in msg.get("images") or []:
                parts.append(Part.from_bytes(data=_pil_jpeg_bytes(img), mime_type="image/jpeg"))
            text = msg.get("text")
            if text:
                parts.append(Part.from_text(text=str(text)))
            if parts:
                contents.append(Content(role="user", parts=parts))
        elif role == "assistant":
            state = msg.get("provider_state")
            if state is not None:
                contents.append(state)
                continue
            parts = []
            if msg.get("text"):
                parts.append(Part.from_text(text=str(msg["text"])))
            for tc in msg.get("tool_calls") or []:
                parts.append(
                    Part.from_function_call(
                        name=str(tc.get("name") or "tool"),
                        args=dict(tc.get("args") or {}),
                    )
                )
            if parts:
                contents.append(Content(role="model", parts=parts))
        elif role == "tool":
            result = msg.get("result")
            if not isinstance(result, dict):
                result = {"result": result}
            # Gemini function responses must be JSON-serializable scalars/lists/dicts.
            safe = json.loads(json.dumps(result, default=str))
            parts = [
                Part.from_function_response(
                    name=str(msg.get("name") or "tool"),
                    response=safe,
                )
            ]
            for img in msg.get("images") or []:
                parts.append(Part.from_bytes(data=_pil_jpeg_bytes(img), mime_type="image/jpeg"))
            caption = msg.get("image_caption")
            if caption and msg.get("images"):
                parts.append(Part.from_text(text=str(caption)))
            contents.append(Content(role="user", parts=parts))
    return contents


async def _gemini_chat_with_tools(
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    user_id: Optional[int] = None,
) -> dict[str, Any]:
    from app.observability.langfuse_client import (
        generation_output_summary,
        observe_generation,
        preview_agent_messages,
    )

    client = get_gemini_client()
    declarations = [
        FunctionDeclaration(
            name=t["name"],
            description=t.get("description") or "",
            parameters_json_schema=t.get("parameters") or {"type": "object", "properties": {}},
        )
        for t in tools
    ]
    config = GenerateContentConfig(
        system_instruction=system or None,
        tools=[Tool(function_declarations=declarations)] if declarations else None,
        thinking_config=ThinkingConfig(thinking_level="low"),
    )
    obs_input = {
        "system_preview": (system or "")[:500],
        "tools": [t.get("name") for t in tools],
        **preview_agent_messages(messages),
    }
    with observe_generation(
        "designer-agent-turn",
        model=model,
        metadata={"agent": "designer", "provider": "gemini"},
        input=obs_input,
        tags=["designer", "gemini"],
    ) as generation:
        response = await client.aio.models.generate_content(
            model=model,
            contents=_gemini_contents_from_messages(messages),
            config=config,
        )
        try:
            from app.admin.usage_helpers import record_from_response
            record_from_response(response, agent="designer", model=model, user_id=user_id)
        except Exception:
            pass

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        provider_state = None
        finish_reason = None
        candidate = (response.candidates or [None])[0]
        if candidate is not None:
            fr = getattr(candidate, "finish_reason", None)
            if fr is not None:
                finish_reason = getattr(fr, "name", None) or str(fr)
            if candidate.content is not None:
                provider_state = candidate.content
                for i, part in enumerate(candidate.content.parts or []):
                    if getattr(part, "text", None) and not getattr(part, "thought", False):
                        text_parts.append(part.text)
                    fc = getattr(part, "function_call", None)
                    if fc is not None and fc.name:
                        tool_calls.append(
                            {
                                "id": getattr(fc, "id", None) or f"{fc.name}:{i}",
                                "name": fc.name,
                                "args": dict(fc.args or {}),
                            }
                        )
        text = "\n".join(text_parts).strip()
        try:
            usage = getattr(response, "usage_metadata", None)
            usage_details = None
            if usage is not None:
                from app.admin.usage_helpers import tokens_from_usage
                inp, out = tokens_from_usage(usage)
                if inp or out:
                    usage_details = {"input": inp, "output": out}
            generation.update(
                output=generation_output_summary(
                    text=text,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                ),
                usage_details=usage_details,
            )
        except Exception:
            pass
    return {
        "text": text,
        "tool_calls": tool_calls,
        "provider_state": provider_state,
        "finish_reason": finish_reason,
    }


def _claude_messages_from_agent(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            blocks: list[dict[str, Any]] = []
            for img in msg.get("images") or []:
                blocks.append(llm_provider.pil_image_block(img))
            if msg.get("text"):
                blocks.append(llm_provider.text_block(str(msg["text"])))
            if blocks:
                out.append({"role": "user", "content": blocks})
        elif role == "assistant":
            state = msg.get("provider_state")
            if isinstance(state, list):
                out.append({"role": "assistant", "content": state})
                continue
            blocks = []
            if msg.get("text"):
                blocks.append({"type": "text", "text": str(msg["text"])})
            for tc in msg.get("tool_calls") or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(tc.get("id") or tc.get("name") or "tool"),
                        "name": str(tc.get("name") or "tool"),
                        "input": dict(tc.get("args") or {}),
                    }
                )
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            result = msg.get("result")
            content = json.dumps(result, default=str) if not isinstance(result, str) else result
            blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": str(msg.get("tool_call_id") or msg.get("name") or "tool"),
                    "content": content[:48000],
                }
            ]
            for img in msg.get("images") or []:
                blocks.append(llm_provider.pil_image_block(img))
            out.append({"role": "user", "content": blocks})
    return out


async def _claude_chat_with_tools(
    *,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    from app.observability.langfuse_client import preview_agent_messages

    claude_tools = [
        {
            "name": t["name"],
            "description": t.get("description") or "",
            "input_schema": t.get("parameters") or {"type": "object", "properties": {}},
        }
        for t in tools
    ]
    msg = await llm_provider.acomplete_message(
        system=system,
        messages=_claude_messages_from_agent(messages),
        tools=claude_tools,
        thinking=True,
        observation_name="designer-agent-turn",
        agent="designer",
        tags=["designer", "claude"],
        observation_input={
            "system_preview": (system or "")[:500],
            "tools": [t.get("name") for t in tools],
            **preview_agent_messages(messages),
        },
    )
    tool_calls = [
        {"id": b.get("id"), "name": b.get("name"), "args": b.get("input") or {}}
        for b in llm_provider.tool_use_blocks(msg)
    ]
    return {
        "text": llm_provider.extract_text(msg),
        "tool_calls": tool_calls,
        "provider_state": llm_provider.serialize_content(msg),
        "finish_reason": getattr(msg, "stop_reason", None),
    }


async def chat_with_tools(
    *,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: Optional[str] = None,
    user_id: Optional[int] = None,
) -> dict[str, Any]:
    """One tool-calling model turn. Returns text + tool_calls + provider_state."""
    if llm_provider.use_claude():
        return await _claude_chat_with_tools(system=system, messages=messages, tools=tools)
    return await _gemini_chat_with_tools(
        model=model or designer_model(),
        system=system,
        messages=messages,
        tools=tools,
        user_id=user_id,
    )


def append_tool_message(
    messages: list[dict[str, Any]],
    *,
    tool_call_id: Any,
    name: str,
    result: dict[str, Any],
    images: Optional[list] = None,
    image_caption: Optional[str] = None,
    max_result_chars: int = 12000,
) -> None:
    """Append a tool result; truncate large JSON payloads for context budget."""
    payload = result
    try:
        raw = json.dumps(result, ensure_ascii=False, default=str)
        if len(raw) > max_result_chars:
            payload = {
                "truncated": True,
                "preview": raw[: max_result_chars // 2],
                "note": f"Result truncated from {len(raw)} chars.",
            }
            if isinstance(result, dict) and result.get("error"):
                payload["error"] = result["error"]
            if isinstance(result, dict) and "generation_index" in result:
                payload["generation_index"] = result["generation_index"]
            if isinstance(result, dict) and "hits" in result:
                hits = result.get("hits") or []
                payload["hit_count"] = len(hits)
                payload["hit_ids"] = [h.get("id") for h in hits[:8] if isinstance(h, dict)]
    except Exception:
        payload = {"error": "failed to serialize tool result"}
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "result": payload,
            "images": list(images or []),
            "image_caption": image_caption,
        }
    )
