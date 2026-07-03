"""Per-model token pricing (USD per 1M tokens) for cost accounting.

Rates sourced from Google Cloud Agent Platform pricing page (Jul 2026) and
Anthropic public pricing. Used by UsageRecorder to compute cost_usd for each
api_usage_event.

Image generation models are token-priced: image output tokens are billed at
the ``image_out`` rate. For a 1K/2K Pro Image that's 1120 tokens * $120/1M
= $0.134 per image.

If a model is unknown, cost is 0 and a warning is logged so pricing gaps are
visible in the admin dashboard rather than silently underreported.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPricing:
    input_per_mtok: float        # text/image input per 1M tokens
    output_per_mtok: float       # text output per 1M tokens
    image_out_per_mtok: float = 0.0   # image output tokens per 1M (0 if N/A)
    audio_in_per_mtok: float = 0.0    # audio input per 1M tokens (Live)
    # cached input tiers omitted for v1; add when needed


# Per 1M tokens, USD
_PRICING: dict[str, ModelPricing] = {
    # Gemini 3.x text
    "gemini-3.5-flash":            ModelPricing(0.50, 3.00),
    "gemini-3-flash-preview":      ModelPricing(0.50, 3.00),
    "gemini-3.1-pro-preview":      ModelPricing(2.00, 12.00),
    "gemini-3-pro-preview":        ModelPricing(2.00, 12.00),
    # Gemini image gen
    "gemini-3-pro-image-preview":  ModelPricing(2.00, 12.00, image_out_per_mtok=120.00),
    "gemini-3-pro-image":          ModelPricing(2.00, 12.00, image_out_per_mtok=120.00),
    "gemini-3.1-flash-image-preview": ModelPricing(0.50, 3.00, image_out_per_mtok=60.00),
    "gemini-3.1-flash-image":      ModelPricing(0.50, 3.00, image_out_per_mtok=60.00),
    "gemini-2.5-flash-image":      ModelPricing(0.30, 2.50, image_out_per_mtok=30.00),
    # Live (voice)
    "gemini-3.1-flash-live-preview": ModelPricing(0.50, 3.00, audio_in_per_mtok=1.00),
    "gemini-2.5-flash-preview-native-audio": ModelPricing(0.50, 3.00, audio_in_per_mtok=1.00),
    # Embeddings
    "gemini-embedding-2-preview":  ModelPricing(0.075, 0.0),
    "text-embedding-004":          ModelPricing(0.025, 0.0),
    # Claude
    "claude-sonnet-4-6":           ModelPricing(3.00, 15.00),
    "claude-sonnet-4-5":           ModelPricing(3.00, 15.00),
    "claude-3-7-sonnet":           ModelPricing(3.00, 15.00),
    "claude-3-5-sonnet":           ModelPricing(3.00, 15.00),
    "claude-opus-4":               ModelPricing(15.00, 75.00),
}

# Aliases / common shorthand used in code
_ALIASES: dict[str, str] = {
    "flash": "gemini-3.1-flash-image-preview",
    "pro": "gemini-3-pro-image-preview",
    "nano-banana-2": "gemini-3.1-flash-image-preview",
    "nano-banana-pro": "gemini-3-pro-image-preview",
}


def resolve_pricing(model: str) -> Optional[ModelPricing]:
    if not model:
        return None
    key = _ALIASES.get(model.lower().strip(), model.strip())
    p = _PRICING.get(key)
    if p is None:
        # try case-insensitive
        for k, v in _PRICING.items():
            if k.lower() == key.lower():
                return v
    return p


def compute_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    image_count: int = 0,
    *,
    image_tokens_per_image: int = 1120,
    audio_input_tokens: int = 0,
) -> float:
    """Return estimated USD cost for a single provider call.

    Image output tokens are estimated as image_count * image_tokens_per_image
    (default 1120 = 1K/2K resolution). When the caller knows the exact token
    count from usage_metadata it should pass output_tokens inclusive of image
    tokens and image_count=0 with a custom rate lookup — but for v1 we use the
    simple per-image estimate which matches Google's documented 1K/2K pricing.
    """
    p = resolve_pricing(model)
    if p is None:
        logger.warning("No pricing entry for model %r — cost recorded as 0", model)
        return 0.0
    cost = 0.0
    cost += (input_tokens / 1_000_000.0) * p.input_per_mtok
    cost += (output_tokens / 1_000_000.0) * p.output_per_mtok
    if image_count and p.image_out_per_mtok:
        cost += (image_count * image_tokens_per_image / 1_000_000.0) * p.image_out_per_mtok
    if audio_input_tokens and p.audio_in_per_mtok:
        cost += (audio_input_tokens / 1_000_000.0) * p.audio_in_per_mtok
    return round(cost, 6)


def list_pricing() -> list[dict]:
    """Return the full pricing table for the admin UI (read-only reference)."""
    out = []
    for model, p in _PRICING.items():
        out.append({
            "model": model,
            "input_per_mtok": p.input_per_mtok,
            "output_per_mtok": p.output_per_mtok,
            "image_out_per_mtok": p.image_out_per_mtok or None,
            "audio_in_per_mtok": p.audio_in_per_mtok or None,
        })
    return out
