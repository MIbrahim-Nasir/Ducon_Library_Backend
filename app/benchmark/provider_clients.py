"""Dev-dashboard LLM provider clients.

These helpers are intentionally small and provider-focused.  They support the
dev benchmark designer agent and provide unit-testable request building for
OpenRouter without changing the production agent globals.
"""
from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Optional

import httpx
from google.genai import types as genai_types
from google.genai.types import GenerateContentConfig, ThinkingConfig
from PIL import Image

from app import llm_provider, prompt_loader
from app.gemini import get_gemini_client, _parse_json_response
from app.benchmark.provider_registry import (
    CLAUDE_NATIVE,
    GEMINI_NATIVE,
    OPENROUTER,
    ModelRouterPair,
)
from app.benchmark.session_config import DesignerSessionConfig


def _pil_data_url(img: Image.Image, *, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _normalize_openrouter_effort(thinking: Optional[str]) -> Optional[str]:
    if not thinking:
        return None
    raw = str(thinking).strip()
    if not raw:
        return None
    mapping = {"High": "high", "disabled": "none", "off": "none"}
    return mapping.get(raw, raw.lower())


def build_openrouter_payload(
    *,
    model_id: str,
    messages: list[dict[str, Any]],
    thinking: Optional[str] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[dict[str, Any]] = None,
    context_compression: Optional[bool] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
    }
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    if response_format:
        payload["response_format"] = response_format
    if context_compression is not None:
        payload["plugins"] = [
            {"id": "context-compression", "enabled": bool(context_compression)}
        ]
    effort = _normalize_openrouter_effort(thinking)
    if effort:
        payload["reasoning"] = {"effort": effort, "exclude": False}
    return payload


def _openrouter_headers() -> dict[str, str]:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_APP_TITLE", "Ducon Dev Benchmark Dashboard")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _json_messages(
    *,
    system: str,
    prompt: str,
    images: Optional[list[Image.Image]] = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images or []:
        content.append({"type": "image_url", "image_url": {"url": _pil_data_url(img)}})
    messages.append({"role": "user", "content": content})
    return messages


async def _openrouter_complete_text(
    *,
    pair: ModelRouterPair,
    system: str,
    prompt: str,
    images: Optional[list[Image.Image]],
    thinking: Optional[str],
    max_tokens: Optional[int],
    as_json: bool,
) -> str:
    payload = build_openrouter_payload(
        model_id=pair.model_id,
        messages=_json_messages(system=system, prompt=prompt, images=images),
        thinking=thinking,
        max_tokens=max_tokens,
        response_format={"type": "json_object"} if as_json else None,
    )
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=_openrouter_headers(),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return str(message.get("content") or "").strip()


async def _gemini_complete_text(
    *,
    pair: ModelRouterPair,
    system: str,
    prompt: str,
    images: Optional[list[Image.Image]],
    thinking: Optional[str],
    as_json: bool,
    max_tokens: Optional[int] = None,
) -> str:
    client = get_gemini_client()
    kwargs: dict[str, Any] = {
        "system_instruction": system or None,
        "response_mime_type": "application/json" if as_json else None,
    }
    if max_tokens:
        kwargs["max_output_tokens"] = int(max_tokens)
    if thinking and thinking.lower() not in {"none", "off", "disabled"}:
        kwargs["thinking_config"] = ThinkingConfig(thinking_level=thinking)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    response = await client.aio.models.generate_content(
        model=pair.model_id,
        contents=[*(images or []), prompt],
        config=GenerateContentConfig(**kwargs),
    )
    return str(response.text or "").strip()


async def _claude_complete_text(
    *,
    pair: ModelRouterPair,
    system: str,
    prompt: str,
    images: Optional[list[Image.Image]],
    thinking: Optional[str],
    max_tokens: Optional[int],
) -> str:
    client = llm_provider.get_async_anthropic_client()
    content = [llm_provider.pil_image_block(img) for img in images or []]
    content.append(llm_provider.text_block(prompt))
    thinking_cfg = _claude_thinking_config(thinking)
    kwargs: dict[str, Any] = {
        "model": pair.model_id,
        "max_tokens": _claude_effective_max_tokens(thinking_cfg, max_tokens),
        "system": system or "",
        "messages": [{"role": "user", "content": content}],
    }
    if thinking_cfg:
        kwargs["thinking"] = thinking_cfg
    msg = await _claude_stream_message(client, kwargs=kwargs)
    return llm_provider.extract_text(msg)


async def complete_text(
    *,
    pair: ModelRouterPair,
    system: str,
    prompt: str,
    images: Optional[list[Image.Image]] = None,
    thinking: Optional[str] = None,
    max_tokens: Optional[int] = None,
    as_json: bool = False,
) -> str:
    prompt_loader.ensure_prompts_loaded()
    if pair.router == GEMINI_NATIVE:
        return await _gemini_complete_text(
            pair=pair,
            system=system,
            prompt=prompt,
            images=images,
            thinking=thinking,
            as_json=as_json,
            max_tokens=max_tokens,
        )
    if pair.router == CLAUDE_NATIVE:
        return await _claude_complete_text(
            pair=pair,
            system=system,
            prompt=prompt,
            images=images,
            thinking=thinking,
            max_tokens=max_tokens,
        )
    if pair.router == OPENROUTER:
        return await _openrouter_complete_text(
            pair=pair,
            system=system,
            prompt=prompt,
            images=images,
            thinking=thinking,
            max_tokens=max_tokens,
            as_json=as_json,
        )
    raise ValueError(f"Unsupported router: {pair.router}")


async def complete_json(
    *,
    pair: ModelRouterPair,
    system: str,
    prompt: str,
    images: Optional[list[Image.Image]] = None,
    thinking: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    text = await complete_text(
        pair=pair,
        system=system,
        prompt=prompt,
        images=images,
        thinking=thinking,
        max_tokens=max_tokens,
        as_json=True,
    )
    if pair.router == GEMINI_NATIVE:
        return _parse_json_response(text)
    try:
        return llm_provider.parse_json_text(text)
    except Exception:
        return json.loads(text)


# ── Tool-calling chat (agentic loop support) ──────────────────────────────────
#
# Provider-agnostic conversation format used by the dev designer agent loop:
#
#   {"role": "user",      "text": str|None, "images": [PIL.Image]|None}
#   {"role": "assistant", "text": str|None,
#                         "tool_calls": [{"id": str, "name": str, "args": dict}]}
#   {"role": "tool",      "tool_call_id": str, "name": str,
#                         "result": dict|str, "images": [PIL.Image]|None}
#
# Tools are declared in a neutral shape:
#   {"name": str, "description": str, "parameters": <JSON schema dict>}
#
# ``chat_with_tools`` performs ONE model call and returns:
#   {"text": str, "tool_calls": [{"id", "name", "args"}]}
# The caller owns the loop (execute tools, append results, call again).


_CLAUDE_THINKING_BUDGET = 4096
# Claude requires max_tokens > thinking.budget_tokens when extended thinking is enabled.
_CLAUDE_MIN_OUTPUT_AFTER_THINKING = 2048


def _claude_thinking_config(thinking: Optional[str]) -> Optional[dict[str, Any]]:
    """Map dev-dashboard thinking modes to Anthropic thinking config."""
    mode = (thinking or "adaptive").strip().lower()
    if mode in {"disabled", "none", "off", ""}:
        return None
    if mode == "adaptive":
        return {"type": "adaptive"}
    # high / medium / max / gemini-style labels → fixed extended-thinking budget
    return {"type": "enabled", "budget_tokens": _CLAUDE_THINKING_BUDGET}


def _claude_effective_max_tokens(
    thinking_cfg: Optional[dict[str, Any]],
    max_tokens: Optional[int],
) -> int:
    requested = int(max_tokens or llm_provider.claude_max_tokens())
    if thinking_cfg and thinking_cfg.get("type") == "enabled":
        budget = int(thinking_cfg.get("budget_tokens") or _CLAUDE_THINKING_BUDGET)
        return max(requested, budget + _CLAUDE_MIN_OUTPUT_AFTER_THINKING)
    return requested


async def _claude_stream_message(
    client: Any,
    *,
    kwargs: dict[str, Any],
    beta_kwargs: Optional[dict[str, Any]] = None,
    beta_betas: Optional[list[str]] = None,
) -> Any:
    """Return the final Message from a streaming Claude API call.

    The Anthropic SDK requires streaming for requests that may exceed 10 minutes
    (common with extended thinking, tool loops, and large max_tokens).
    """
    if beta_kwargs is not None and hasattr(client, "beta") and hasattr(client.beta, "messages"):
        try:
            async with client.beta.messages.stream(
                betas=beta_betas or [],
                **beta_kwargs,
            ) as stream:
                return await stream.get_final_message()
        except Exception:
            pass
    async with client.messages.stream(**kwargs) as stream:
        return await stream.get_final_message()


def _claude_tool_result_block(msg: dict[str, Any]) -> dict[str, Any]:
    """Build one Anthropic tool_result block, with optional image attachments."""
    result = msg.get("result")
    name = str(msg.get("name") or "tool")
    parts: list[dict[str, Any]] = []
    caption = str(msg.get("image_caption") or "").strip()
    if caption:
        parts.append(llm_provider.text_block(caption))
    parts.append(
        llm_provider.text_block(
            json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
        )
    )
    for img in msg.get("images") or []:
        parts.append(llm_provider.pil_image_block(img))
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": str(msg.get("tool_call_id") or msg.get("name") or "tool"),
        "content": parts,
    }
    if isinstance(result, dict) and result.get("error"):
        block["is_error"] = True
    return block


def _pil_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _gemini_contents_from_messages(messages: list[dict[str, Any]]) -> list[Any]:
    contents: list[Any] = []
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            parts: list[Any] = []
            if msg.get("text"):
                parts.append(genai_types.Part.from_text(text=str(msg["text"])))
            for img in msg.get("images") or []:
                parts.append(
                    genai_types.Part.from_bytes(data=_pil_png_bytes(img), mime_type="image/jpeg")
                )
            if parts:
                contents.append(genai_types.Content(role="user", parts=parts))
        elif role == "assistant":
            # Gemini 3 requires thought_signature to be replayed on function-call
            # parts; the raw model Content (stashed as provider_state) carries it.
            state = msg.get("provider_state")
            if isinstance(state, genai_types.Content):
                contents.append(state)
                continue
            parts = []
            if msg.get("text"):
                parts.append(genai_types.Part.from_text(text=str(msg["text"])))
            for tc in msg.get("tool_calls") or []:
                parts.append(
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            name=tc["name"], args=tc.get("args") or {}
                        )
                    )
                )
            if parts:
                contents.append(genai_types.Content(role="model", parts=parts))
        elif role == "tool":
            result = msg.get("result")
            if not isinstance(result, dict):
                result = {"result": result}
            parts = [
                genai_types.Part.from_function_response(
                    name=str(msg.get("name") or "tool"), response=result
                )
            ]
            for img in msg.get("images") or []:
                parts.append(
                    genai_types.Part.from_bytes(data=_pil_png_bytes(img), mime_type="image/jpeg")
                )
            contents.append(genai_types.Content(role="user", parts=parts))
    return contents


async def _gemini_chat_with_tools(
    *,
    pair: ModelRouterPair,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    thinking: Optional[str],
    max_tokens: Optional[int],
) -> dict[str, Any]:
    client = get_gemini_client()
    declarations = [
        genai_types.FunctionDeclaration(
            name=t["name"],
            description=t.get("description") or "",
            parameters_json_schema=t.get("parameters") or {"type": "object", "properties": {}},
        )
        for t in tools
    ]
    kwargs: dict[str, Any] = {
        "system_instruction": system or None,
        "tools": [genai_types.Tool(function_declarations=declarations)] if declarations else None,
    }
    if max_tokens:
        kwargs["max_output_tokens"] = int(max_tokens)
    if thinking and thinking.lower() not in {"none", "off", "disabled"}:
        kwargs["thinking_config"] = ThinkingConfig(thinking_level=thinking)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    response = await client.aio.models.generate_content(
        model=pair.model_id,
        contents=_gemini_contents_from_messages(messages),
        config=GenerateContentConfig(**kwargs),
    )
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    provider_state = None
    candidate = (response.candidates or [None])[0]
    finish_reason = None
    if candidate is not None:
        fr = getattr(candidate, "finish_reason", None)
        if fr is not None:
            finish_reason = getattr(fr, "name", None) or str(fr)
    if candidate is not None and candidate.content is not None:
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
    return {
        "text": "\n".join(text_parts).strip(),
        "tool_calls": tool_calls,
        "provider_state": provider_state,
        "finish_reason": finish_reason,
    }


def _openrouter_messages_with_tools(
    *, system: str, messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            content: list[dict[str, Any]] = []
            if msg.get("text"):
                content.append({"type": "text", "text": str(msg["text"])})
            for img in msg.get("images") or []:
                content.append({"type": "image_url", "image_url": {"url": _pil_data_url(img)}})
            if content:
                out.append({"role": "user", "content": content})
        elif role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": msg.get("text") or ""}
            tcs = []
            for tc in msg.get("tool_calls") or []:
                tcs.append(
                    {
                        "id": str(tc.get("id") or tc["name"]),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("args") or {}, ensure_ascii=False),
                        },
                    }
                )
            if tcs:
                entry["tool_calls"] = tcs
            out.append(entry)
        elif role == "tool":
            result = msg.get("result")
            content_str = (
                json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
            )
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": str(msg.get("tool_call_id") or msg.get("name") or "tool"),
                    "content": content_str,
                }
            )
            # OpenAI-format tool results are text-only; feed images back as a
            # follow-up user message so the model can still see them.
            imgs = msg.get("images") or []
            if imgs:
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"[Image output from tool {msg.get('name')}]"},
                            *(
                                {"type": "image_url", "image_url": {"url": _pil_data_url(img)}}
                                for img in imgs
                            ),
                        ],
                    }
                )
    return out


async def _openrouter_chat_with_tools(
    *,
    pair: ModelRouterPair,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    thinking: Optional[str],
    max_tokens: Optional[int],
    session_config: Optional[DesignerSessionConfig] = None,
) -> dict[str, Any]:
    compression = None
    if session_config is not None:
        compression = session_config.openrouter_compression_enabled()
    payload = build_openrouter_payload(
        model_id=pair.model_id,
        messages=_openrouter_messages_with_tools(system=system, messages=messages),
        thinking=thinking,
        max_tokens=max_tokens,
        context_compression=compression,
    )
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description") or "",
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=_openrouter_headers(),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls: list[dict[str, Any]] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(
            {
                "id": str(tc.get("id") or fn.get("name") or "tool"),
                "name": str(fn.get("name") or ""),
                "args": args if isinstance(args, dict) else {},
            }
        )
    content = message.get("content")
    if isinstance(content, list):
        content = " ".join(
            str(c.get("text") or "") for c in content if isinstance(c, dict)
        )
    return {
        "text": str(content or "").strip(),
        "tool_calls": tool_calls,
        "finish_reason": choice.get("finish_reason"),
    }


def _claude_messages_with_tools(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")
        if role == "user":
            content: list[dict[str, Any]] = []
            for img in msg.get("images") or []:
                content.append(llm_provider.pil_image_block(img))
            if msg.get("text"):
                content.append(llm_provider.text_block(str(msg["text"])))
            if content:
                out.append({"role": "user", "content": content})
            i += 1
            continue
        if role == "assistant":
            state = msg.get("provider_state")
            if isinstance(state, list) and state:
                out.append({"role": "assistant", "content": state})
                i += 1
                continue
            content = []
            if msg.get("text"):
                content.append(llm_provider.text_block(str(msg["text"])))
            for tc in msg.get("tool_calls") or []:
                content.append(
                    {
                        "type": "tool_use",
                        "id": str(tc.get("id") or tc["name"]),
                        "name": tc["name"],
                        "input": tc.get("args") or {},
                    }
                )
            if content:
                out.append({"role": "assistant", "content": content})
            i += 1
            continue
        if role == "tool":
            # Anthropic requires ALL tool_result blocks for one assistant turn in a
            # single user message, with tool_result blocks first in content.
            tool_blocks: list[dict[str, Any]] = []
            while i < len(messages) and messages[i].get("role") == "tool":
                tool_blocks.append(_claude_tool_result_block(messages[i]))
                i += 1
            if tool_blocks:
                out.append({"role": "user", "content": tool_blocks})
            continue
        i += 1
    return out


async def _claude_chat_with_tools(
    *,
    pair: ModelRouterPair,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    thinking: Optional[str],
    max_tokens: Optional[int],
    session_config: Optional[DesignerSessionConfig] = None,
) -> dict[str, Any]:
    client = llm_provider.get_async_anthropic_client()
    thinking_cfg = _claude_thinking_config(thinking)
    kwargs: dict[str, Any] = {
        "model": pair.model_id,
        "max_tokens": _claude_effective_max_tokens(thinking_cfg, max_tokens),
        "system": system or "",
        "messages": _claude_messages_with_tools(messages),
    }
    if tools:
        kwargs["tools"] = [
            {
                "name": t["name"],
                "description": t.get("description") or "",
                "input_schema": t.get("parameters") or {"type": "object", "properties": {}},
            }
            for t in tools
        ]
    if thinking_cfg:
        kwargs["thinking"] = thinking_cfg

    use_compaction = (
        session_config is not None
        and session_config.claude_compaction_enabled(router=CLAUDE_NATIVE)
    )
    compaction_edit: dict[str, Any] | None = None
    if use_compaction:
        compaction_edit = {
            "type": "compact_20260112",
            "trigger": {
                "type": "input_tokens",
                "value": session_config.claude_compaction_trigger_tokens,
            },
        }
        instructions = session_config.claude_compaction_instructions
        if instructions:
            compaction_edit["instructions"] = instructions

    beta_kwargs = None
    if compaction_edit is not None:
        beta_kwargs = {
            **kwargs,
            "context_management": {"edits": [compaction_edit]},
        }

    # context_management is beta-only; never pass it to the stable Messages API.
    msg = await _claude_stream_message(
        client,
        kwargs=kwargs,
        beta_kwargs=beta_kwargs,
        beta_betas=["compact-2026-01-12"] if beta_kwargs else None,
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
    pair: ModelRouterPair,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    thinking: Optional[str] = None,
    max_tokens: Optional[int] = None,
    session_config: Optional[DesignerSessionConfig] = None,
) -> dict[str, Any]:
    """One tool-calling model turn. Returns {"text", "tool_calls", "finish_reason", ...}."""
    prompt_loader.ensure_prompts_loaded()
    if pair.router == GEMINI_NATIVE:
        return await _gemini_chat_with_tools(
            pair=pair, system=system, messages=messages, tools=tools,
            thinking=thinking, max_tokens=max_tokens,
        )
    if pair.router == CLAUDE_NATIVE:
        return await _claude_chat_with_tools(
            pair=pair, system=system, messages=messages, tools=tools,
            thinking=thinking, max_tokens=max_tokens,
            session_config=session_config,
        )
    if pair.router == OPENROUTER:
        return await _openrouter_chat_with_tools(
            pair=pair, system=system, messages=messages, tools=tools,
            thinking=thinking, max_tokens=max_tokens,
            session_config=session_config,
        )
    raise ValueError(f"Unsupported router: {pair.router}")
