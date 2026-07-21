"""Shared helpers to record AI usage from provider responses (non-blocking)."""
from __future__ import annotations

from typing import Any, Optional


def _extract_usage_dict(usage: Any) -> dict:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    if isinstance(usage, dict):
        return usage
    return {}


def tokens_from_usage(usage: Any) -> tuple[int, int]:
    u = _extract_usage_dict(usage)
    inp = int(u.get("input_tokens") or u.get("prompt_token_count") or u.get("prompt_tokens") or 0)
    out = int(u.get("output_tokens") or u.get("candidates_token_count") or u.get("completion_tokens") or 0)
    return inp, out


def record_from_response(
    response: Any,
    *,
    agent: str,
    model: str,
    provider: str = "gemini",
    user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None,
    image_count: int = 0,
    latency_ms: Optional[int] = None,
    status: str = "success",
    error_message: Optional[str] = None,
) -> None:
    usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
    inp, out = tokens_from_usage(usage)
    try:
        from app.admin.usage_recorder import record
        record(
            agent=agent,
            model=model,
            provider=provider,
            user_id=user_id,
            guest_session_id=guest_session_id,
            input_tokens=inp,
            output_tokens=out,
            image_count=image_count,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
        )
    except Exception:
        pass


def record_from_usage_dict(
    usage: Any,
    *,
    agent: str,
    model: str,
    provider: str = "gemini",
    user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None,
    image_count: int = 0,
    latency_ms: Optional[int] = None,
    status: str = "success",
    error_message: Optional[str] = None,
) -> None:
    inp, out = tokens_from_usage(usage)
    try:
        from app.admin.usage_recorder import record
        record(
            agent=agent,
            model=model,
            provider=provider,
            user_id=user_id,
            guest_session_id=guest_session_id,
            input_tokens=inp,
            output_tokens=out,
            image_count=image_count,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
        )
    except Exception:
        pass
