"""Provider/router model registry for the dev benchmark dashboard.

The dashboard stores a model together with the router used to call it.  This
keeps "gemini-3-flash-preview via Gemini native" distinct from the same logical
model routed through OpenRouter or any future gateway.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import httpx

from app.admin.settings_catalog import AI_MODELS


RouterId = str

GEMINI_NATIVE: RouterId = "gemini_native"
CLAUDE_NATIVE: RouterId = "claude_native"
OPENROUTER: RouterId = "openrouter"

ROUTERS = [
    {
        "id": GEMINI_NATIVE,
        "label": "Gemini native",
        "env_key": "GOOGLE_API_KEY",
        "supports_images": True,
    },
    {
        "id": CLAUDE_NATIVE,
        "label": "Claude native SDK",
        "env_key": "ANTHROPIC_API_KEY",
        "supports_images": True,
    },
    {
        "id": OPENROUTER,
        "label": "OpenRouter",
        "env_key": "OPENROUTER_API_KEY",
        "supports_images": True,
    },
]

GEMINI_THINKING = ["minimal", "low", "medium", "High", "max"]
CLAUDE_THINKING = ["disabled", "adaptive", "high"]
OPENROUTER_REASONING = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
NO_THINKING = ["none"]

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_REGISTRY_FILE = _REPO_ROOT / "benchmark_data" / "model_pairs.json"


@dataclass(frozen=True)
class ModelRouterPair:
    id: str
    router: RouterId
    model_id: str
    label: str
    roles: list[str]
    thinking_modes: list[str]
    source: str = "builtin"
    supports_reasoning: bool = False


def pair_id(router: str, model_id: str) -> str:
    return f"{router}:{model_id.strip()}"


def _roles_for(router: str, model_id: str) -> list[str]:
    roles = ["prompt", "designer"]
    if router == GEMINI_NATIVE and "image" in model_id.lower():
        roles.insert(0, "image")
    return roles


def _looks_reasoning_capable(model_id: str) -> bool:
    mid = model_id.lower()
    markers = (
        "reason",
        "thinking",
        "deepseek-r1",
        "qwen3",
        "/o1",
        "/o3",
        "/o4",
        "gpt-5",
        "claude-3.7",
        "claude-sonnet-4",
        "claude-opus-4",
    )
    return any(m in mid for m in markers)


def infer_thinking_modes(
    router: str,
    model_id: str,
    *,
    openrouter_metadata: Optional[dict[str, Any]] = None,
) -> tuple[list[str], bool]:
    """Return UI-safe thinking choices for a router/model pair.

    For OpenRouter we default to "none" unless the model metadata or a conservative
    name heuristic says reasoning is supported.  This avoids showing unsupported
    effort values that can trigger API validation errors.
    """
    if router == GEMINI_NATIVE:
        return list(GEMINI_THINKING), True
    if router == CLAUDE_NATIVE:
        return list(CLAUDE_THINKING), True
    if router != OPENROUTER:
        return list(NO_THINKING), False

    reasoning = (openrouter_metadata or {}).get("reasoning")
    if isinstance(reasoning, dict):
        supported = (
            reasoning.get("supported_efforts")
            or reasoning.get("efforts")
            or reasoning.get("allowed_efforts")
        )
        if isinstance(supported, list) and supported:
            modes = ["none", *[str(x) for x in supported if str(x) != "none"]]
            return _dedupe_modes(modes), True
        if reasoning.get("supports_max_tokens") or reasoning.get("default_effort"):
            return list(OPENROUTER_REASONING), True
    if _looks_reasoning_capable(model_id):
        return list(OPENROUTER_REASONING), True
    return list(NO_THINKING), False


def _dedupe_modes(modes: list[str]) -> list[str]:
    out: list[str] = []
    for mode in modes:
        if mode not in out:
            out.append(mode)
    return out or list(NO_THINKING)


def make_pair(
    *,
    router: str,
    model_id: str,
    label: Optional[str] = None,
    roles: Optional[list[str]] = None,
    thinking_modes: Optional[list[str]] = None,
    source: str = "custom",
    openrouter_metadata: Optional[dict[str, Any]] = None,
) -> ModelRouterPair:
    router = (router or GEMINI_NATIVE).strip()
    model_id = (model_id or "").strip()
    if not model_id:
        raise ValueError("model_id is required")
    if router not in {GEMINI_NATIVE, CLAUDE_NATIVE, OPENROUTER}:
        raise ValueError(f"Unsupported router: {router}")
    inferred_modes, supports_reasoning = infer_thinking_modes(
        router, model_id, openrouter_metadata=openrouter_metadata
    )
    modes = _dedupe_modes(thinking_modes or inferred_modes)
    return ModelRouterPair(
        id=pair_id(router, model_id),
        router=router,
        model_id=model_id,
        label=label or f"{model_id} via {router.replace('_', ' ')}",
        roles=roles or _roles_for(router, model_id),
        thinking_modes=modes,
        source=source,
        supports_reasoning=supports_reasoning or any(m != "none" for m in modes),
    )


def _builtin_model_ids() -> set[str]:
    ids: set[str] = {
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3.5-flash",
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-image-preview",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-live-preview",
    }
    for spec in AI_MODELS.settings:
        if spec.value_type == "string" and spec.key.endswith("MODEL"):
            val = spec.default
            if isinstance(val, str) and val.strip():
                ids.add(val.strip())
    return ids


def builtin_pairs() -> list[ModelRouterPair]:
    pairs = [
        make_pair(router=GEMINI_NATIVE, model_id=mid, source="builtin")
        for mid in sorted(_builtin_model_ids())
        if mid.startswith("gemini")
    ]
    pairs.extend(
        [
            make_pair(router=CLAUDE_NATIVE, model_id="claude-sonnet-4-6", source="builtin"),
            make_pair(router=CLAUDE_NATIVE, model_id="claude-opus-4-6", source="builtin"),
        ]
    )
    return pairs


def _read_custom_pairs_sync() -> list[dict[str, Any]]:
    if not _REGISTRY_FILE.exists():
        return []
    with open(_REGISTRY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _write_custom_pairs_sync(items: list[dict[str, Any]]) -> None:
    _REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _REGISTRY_FILE.with_suffix(f".{uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _REGISTRY_FILE)


async def list_custom_pairs() -> list[ModelRouterPair]:
    raw_items = await asyncio.to_thread(_read_custom_pairs_sync)
    pairs: list[ModelRouterPair] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        try:
            pairs.append(
                make_pair(
                    router=str(raw.get("router") or ""),
                    model_id=str(raw.get("model_id") or ""),
                    label=raw.get("label"),
                    roles=raw.get("roles") if isinstance(raw.get("roles"), list) else None,
                    thinking_modes=(
                        raw.get("thinking_modes")
                        if isinstance(raw.get("thinking_modes"), list)
                        else None
                    ),
                    source="custom",
                )
            )
        except ValueError:
            continue
    return pairs


async def list_model_pairs() -> list[ModelRouterPair]:
    by_id = {p.id: p for p in builtin_pairs()}
    for pair in await list_custom_pairs():
        by_id[pair.id] = pair
    return sorted(by_id.values(), key=lambda p: (p.router, p.model_id))


async def get_model_pair(ref: Any) -> ModelRouterPair:
    if isinstance(ref, dict):
        router = str(ref.get("router") or "")
        model_id = str(ref.get("model_id") or "")
        if router and model_id:
            return make_pair(
                router=router,
                model_id=model_id,
                label=ref.get("label"),
                thinking_modes=(
                    ref.get("thinking_modes")
                    if isinstance(ref.get("thinking_modes"), list)
                    else None
                ),
                source=str(ref.get("source") or "custom"),
            )
        ref = ref.get("id")

    ref_str = str(ref or "").strip()
    pairs = await list_model_pairs()
    for pair in pairs:
        if pair.id == ref_str:
            return pair
    if ":" in ref_str:
        router, model_id = ref_str.split(":", 1)
        return make_pair(router=router, model_id=model_id)
    return make_pair(router=GEMINI_NATIVE, model_id=ref_str)


async def save_model_pair(payload: dict[str, Any]) -> ModelRouterPair:
    metadata: Optional[dict[str, Any]] = None
    router = str(payload.get("router") or "").strip()
    model_id = str(payload.get("model_id") or "").strip()
    if router == OPENROUTER:
        metadata = await fetch_openrouter_model_metadata(model_id)

    pair = make_pair(
        router=router,
        model_id=model_id,
        label=payload.get("label"),
        roles=payload.get("roles") if isinstance(payload.get("roles"), list) else None,
        thinking_modes=(
            payload.get("thinking_modes")
            if isinstance(payload.get("thinking_modes"), list)
            else None
        ),
        source="custom",
        openrouter_metadata=metadata,
    )
    raw_items = await asyncio.to_thread(_read_custom_pairs_sync)
    by_id = {str(item.get("id") or pair_id(item.get("router", ""), item.get("model_id", ""))): item for item in raw_items if isinstance(item, dict)}
    by_id[pair.id] = asdict(pair)
    await asyncio.to_thread(_write_custom_pairs_sync, list(by_id.values()))
    return pair


async def delete_model_pair(pair_id_value: str) -> bool:
    raw_items = await asyncio.to_thread(_read_custom_pairs_sync)
    next_items = [
        item
        for item in raw_items
        if isinstance(item, dict)
        and str(item.get("id") or pair_id(item.get("router", ""), item.get("model_id", "")))
        != pair_id_value
    ]
    if len(next_items) == len(raw_items):
        return False
    await asyncio.to_thread(_write_custom_pairs_sync, next_items)
    return True


async def fetch_openrouter_model_metadata(model_id: str) -> Optional[dict[str, Any]]:
    """Best-effort model metadata fetch.

    OpenRouter documents that GET /api/v1/models includes per-model reasoning
    metadata.  The endpoint is public enough for discovery in most setups; if it
    is unavailable we fall back to conservative local inference.
    """
    model_id = (model_id or "").strip()
    if not model_id:
        return None
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get("https://openrouter.ai/api/v1/models")
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return None

    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return None
    for item in items:
        if isinstance(item, dict) and item.get("id") == model_id:
            return item
    return None


def serialize_pair(pair: ModelRouterPair) -> dict[str, Any]:
    return asdict(pair)
