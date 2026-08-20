"""
Langfuse LLM observability — optional, fail-open.

Aligned with Langfuse skill / best-practices (good names, generations with
model+tokens, session_id / user_id / tags, nested observations, truncated I/O).

When ``LANGFUSE_ENABLED`` is false / unset, or keys are missing, every helper
is a no-op. Dual-write from ``app.admin.usage_recorder.record`` covers paths
without an active ``observe_generation``; those dual-writes always set a
structured input/output summary so the UI never shows null I/O.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# When True, ``record_generation`` is a no-op (active ``observe_generation`` owns the span).
_suppress_dual_write: ContextVar[bool] = ContextVar("langfuse_suppress_dual_write", default=False)

_client: Any = None
_init_attempted: bool = False
_MAX_PREVIEW = 2000
DEFAULT_SERVICE_NAME = "ducon-library-backend"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _keys_present() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def is_enabled() -> bool:
    """True only when explicitly enabled and both API keys are set."""
    try:
        from app.admin.settings_store import cfg_bool
        enabled = cfg_bool("LANGFUSE_ENABLED", False)
    except Exception:
        enabled = _env_bool("LANGFUSE_ENABLED", False)
    if not enabled:
        return False
    return _keys_present()


def _base_url() -> Optional[str]:
    return (
        os.getenv("LANGFUSE_BASE_URL")
        or os.getenv("LANGFUSE_HOST")
        or None
    )


def _ensure_service_name() -> str:
    """
    OTEL Resource defaults to ``unknown_service`` unless ``OTEL_SERVICE_NAME``
    is set. Prefer LANGFUSE_SERVICE_NAME, then OTEL_SERVICE_NAME, then default.
    """
    name = (
        (os.getenv("LANGFUSE_SERVICE_NAME") or "").strip()
        or (os.getenv("OTEL_SERVICE_NAME") or "").strip()
        or DEFAULT_SERVICE_NAME
    )
    # Set before Langfuse builds its TracerProvider / Resource.
    if not (os.getenv("OTEL_SERVICE_NAME") or "").strip():
        os.environ["OTEL_SERVICE_NAME"] = name
    return name


def get_client() -> Any:
    """Lazy Langfuse client. Returns None when disabled or init fails.

    Import happens only after env is loaded (uvicorn/dotenv already applied).
    """
    global _client, _init_attempted
    if not is_enabled():
        return None
    if _client is not None:
        return _client
    if _init_attempted:
        return None
    _init_attempted = True
    try:
        _ensure_service_name()
        from langfuse import Langfuse

        kwargs: dict[str, Any] = {
            "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
            "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
        }
        base = _base_url()
        if base:
            kwargs["base_url"] = base
        _client = Langfuse(**kwargs)
        return _client
    except Exception:
        logger.warning("Langfuse client init failed — observability disabled", exc_info=True)
        _client = None
        return None


def _truncate(value: Any, limit: int = _MAX_PREVIEW) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, default=str, ensure_ascii=False)
        except Exception:
            text = str(value)
        if len(text) <= limit:
            return value
        return text[:limit] + "…"
    return value


def _user_str(user_id: Optional[int] = None) -> Optional[str]:
    if user_id is None:
        return None
    return str(user_id)


def preview_user_text(parts: Any) -> Optional[str]:
    """Extract a short user-facing text preview from chat input parts (no media blobs)."""
    if parts is None:
        return None
    if isinstance(parts, str):
        return _truncate(parts.strip()) or None
    texts: list[str] = []
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, str) and part.strip():
                texts.append(part.strip())
            elif isinstance(part, dict) and part.get("type") == "text":
                t = (part.get("text") or "").strip()
                if t:
                    texts.append(t)
    if not texts:
        return None
    return _truncate("\n".join(texts))


def preview_agent_messages(
    messages: list[dict[str, Any]] | None,
    *,
    max_messages: int = 8,
    text_limit: int = 400,
) -> dict[str, Any]:
    """Compact designer/chat agent transcript preview (no image bytes)."""
    msgs = list(messages or [])
    tail = msgs[-max_messages:] if len(msgs) > max_messages else msgs
    preview: list[dict[str, Any]] = []
    for msg in tail:
        role = msg.get("role") or "unknown"
        entry: dict[str, Any] = {"role": role}
        text = msg.get("text")
        if text:
            entry["text"] = _truncate(str(text), text_limit)
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            entry["tool_calls"] = [
                {
                    "name": tc.get("name"),
                    "args": _truncate(tc.get("args") or {}, 300),
                }
                for tc in tool_calls[:6]
            ]
        if role == "tool":
            entry["name"] = msg.get("name")
            entry["result"] = _truncate(msg.get("result"), 400)
        images = msg.get("images") or []
        if images:
            entry["image_count"] = len(images)
        content = msg.get("content")
        if content is not None and "text" not in entry:
            if isinstance(content, str):
                entry["text"] = _truncate(content, text_limit)
            elif isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = (block.get("text") or "").strip()
                        if t:
                            texts.append(t)
                    elif isinstance(block, str) and block.strip():
                        texts.append(block.strip())
                if texts:
                    entry["text"] = _truncate("\n".join(texts), text_limit)
                entry["content_blocks"] = len(content)
        preview.append(entry)
    return {
        "message_count": len(msgs),
        "messages": preview,
        "truncated": len(msgs) > len(tail),
    }


def generation_output_summary(
    *,
    text: Optional[str] = None,
    tool_calls: Optional[list[dict[str, Any]]] = None,
    finish_reason: Any = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Structured generation output for Langfuse (never leave blank on success)."""
    out: dict[str, Any] = {}
    if text is not None:
        out["text"] = _truncate(str(text), 1500)
    if tool_calls:
        out["tool_calls"] = [
            {
                "name": tc.get("name"),
                "id": tc.get("id"),
                "args": _truncate(tc.get("args") or {}, 400),
            }
            for tc in tool_calls[:12]
        ]
        out["tool_call_count"] = len(tool_calls)
    if finish_reason is not None:
        out["finish_reason"] = str(finish_reason)
    if extra:
        out.update(extra)
    if not out:
        out["status"] = "ok"
    return out


class _NoOpObservation:
    """Stand-in when Langfuse is disabled."""

    def update(self, **kwargs: Any) -> None:
        return None

    def end(self) -> None:
        return None

    def __enter__(self) -> "_NoOpObservation":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


@contextmanager
def start_trace(
    name: str,
    *,
    user_id: Optional[int] = None,
    session_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    input: Any = None,
    as_type: str = "span",
) -> Iterator[Any]:
    """
    Open a root observation (one unit of work / chat turn / agent run).

    Sets ``user_id``, ``session_id``, and ``tags`` via ``propagate_attributes``
    so Sessions / Users / dashboards work. Yields a no-op when disabled.
    """
    client = get_client()
    if client is None:
        yield _NoOpObservation()
        return

    meta = dict(metadata or {})
    if user_id is not None:
        meta.setdefault("user_id", user_id)
    truncated_input = _truncate(input)

    try:
        cm = client.start_as_current_observation(
            as_type=as_type,
            name=name,
            input=truncated_input,
            metadata=meta or None,
        )
    except Exception:
        logger.debug("Langfuse start_trace open failed", exc_info=True)
        yield _NoOpObservation()
        return

    with cm as span:
        if truncated_input is not None:
            try:
                span.update(input=truncated_input)
            except Exception:
                pass

        attrs: dict[str, Any] = {}
        uid = _user_str(user_id)
        if uid:
            attrs["user_id"] = uid
        if session_id:
            attrs["session_id"] = str(session_id)
        if tags:
            attrs["tags"] = list(tags)
        if meta:
            attrs["metadata"] = meta

        if attrs:
            try:
                from langfuse import propagate_attributes

                with propagate_attributes(**attrs):
                    yield span
                    return
            except Exception:
                logger.debug("Langfuse propagate_attributes skipped", exc_info=True)
        yield span


def update_trace_output(output: Any) -> None:
    """Set root / current observation output (assistant reply preview). Fail-open."""
    client = get_client()
    if client is None:
        return
    try:
        truncated = _truncate(output)
        try:
            client.update_current_span(output=truncated)
        except Exception:
            pass
    except Exception:
        logger.debug("Langfuse update_trace_output failed", exc_info=True)


@contextmanager
def observe_generation(
    name: str,
    *,
    model: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    input: Any = None,
    output: Any = None,
    tags: Optional[list[str]] = None,
    suppress_dual_write: bool = True,
) -> Iterator[Any]:
    """
    Nested ``generation`` observation around an LLM call (model + tokens).

    Always sets ``input`` when provided. Callers should ``.update(output=...)``
    before exit; if ``output`` is passed here it is applied on successful exit
    when the observation was not already updated with output.

    When ``suppress_dual_write`` is True (default), concurrent
    ``record_generation`` calls in this context are skipped.
    """
    client = get_client()
    if client is None:
        yield _NoOpObservation()
        return

    token = _suppress_dual_write.set(True) if suppress_dual_write else None
    t0 = time.perf_counter()
    meta = dict(metadata or {})
    truncated_input = _truncate(input)
    try:
        try:
            cm = client.start_as_current_observation(
                as_type="generation",
                name=name,
                model=model,
                input=truncated_input,
                output=_truncate(output) if output is not None else None,
                metadata=meta or None,
            )
        except Exception:
            logger.debug("Langfuse observe_generation open failed", exc_info=True)
            yield _NoOpObservation()
            return

        prop_cm = None
        if tags:
            try:
                from langfuse import propagate_attributes

                prop_cm = propagate_attributes(tags=list(tags))
            except Exception:
                prop_cm = None

        if prop_cm is not None:
            prop_cm.__enter__()
        try:
            with cm as generation:
                try:
                    yield generation
                except Exception as exc:
                    try:
                        generation.update(
                            level="ERROR",
                            status_message=str(exc)[:500],
                            output=_truncate({"error": str(exc)[:500]}),
                            metadata={
                                **meta,
                                "latency_ms": int((time.perf_counter() - t0) * 1000),
                            },
                        )
                    except Exception:
                        pass
                    raise
                try:
                    generation.update(
                        metadata={
                            **meta,
                            "latency_ms": int((time.perf_counter() - t0) * 1000),
                        },
                    )
                except Exception:
                    pass
        finally:
            if prop_cm is not None:
                try:
                    prop_cm.__exit__(None, None, None)
                except Exception:
                    pass
    finally:
        if token is not None:
            _suppress_dual_write.reset(token)


@contextmanager
def observe_span(
    name: str,
    *,
    as_type: str = "span",
    metadata: Optional[dict[str, Any]] = None,
    input: Any = None,
    tags: Optional[list[str]] = None,
) -> Iterator[Any]:
    """Nested span / tool / embedding observation. Fail-open."""
    client = get_client()
    if client is None:
        yield _NoOpObservation()
        return

    meta = dict(metadata or {})
    truncated_input = _truncate(input)
    try:
        cm = client.start_as_current_observation(
            as_type=as_type,
            name=name,
            input=truncated_input,
            metadata=meta or None,
        )
    except Exception:
        logger.debug("Langfuse observe_span open failed", exc_info=True)
        yield _NoOpObservation()
        return

    prop_cm = None
    if tags:
        try:
            from langfuse import propagate_attributes

            prop_cm = propagate_attributes(tags=list(tags))
        except Exception:
            prop_cm = None

    if prop_cm is not None:
        prop_cm.__enter__()
    try:
        with cm as span:
            try:
                yield span
            except Exception as exc:
                try:
                    span.update(
                        level="ERROR",
                        status_message=str(exc)[:500],
                        output=_truncate({"error": str(exc)[:500]}),
                    )
                except Exception:
                    pass
                raise
    finally:
        if prop_cm is not None:
            try:
                prop_cm.__exit__(None, None, None)
            except Exception:
                pass


def record_generation(
    *,
    name: Optional[str] = None,
    agent: str = "unknown",
    model: str = "",
    provider: str = "gemini",
    input_tokens: int = 0,
    output_tokens: int = 0,
    image_count: int = 0,
    latency_ms: Optional[int] = None,
    user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None,
    status: str = "success",
    error_message: Optional[str] = None,
    cost_usd: Optional[float] = None,
    tags: Optional[list[str]] = None,
    input: Any = None,
    output: Any = None,
) -> None:
    """
    Fire-and-forget generation (post-hoc from usage sinks).

    Nests under the current observation when one is active (e.g. chat-response).
    Always sets input + output (structured summary when callers omit them) so
    Langfuse never shows null/undefined I/O on success.
    Never raises. No-op when disabled or dual-write suppressed.
    """
    if _suppress_dual_write.get():
        return
    client = get_client()
    if client is None:
        return
    try:
        meta: dict[str, Any] = {
            "agent": agent,
            "provider": provider,
            "status": status,
        }
        if user_id is not None:
            meta["user_id"] = user_id
        if guest_session_id:
            meta["guest_session_id"] = guest_session_id
        if latency_ms is not None:
            meta["latency_ms"] = latency_ms
        if image_count:
            meta["image_count"] = image_count
        if cost_usd is not None:
            meta["cost_usd"] = cost_usd
        if error_message:
            meta["error_message"] = str(error_message)[:500]

        usage_details: dict[str, int] = {}
        if input_tokens:
            usage_details["input"] = int(input_tokens)
        if output_tokens:
            usage_details["output"] = int(output_tokens)

        cost_details: Optional[dict[str, float]] = None
        if cost_usd is not None:
            cost_details = {"total": float(cost_usd)}

        # Stable low-cardinality name (verb-style); model stays on the generation field.
        obs_name = name or f"{agent.replace('_', '-')}-generation"
        feature_tags = list(tags) if tags else [agent.replace("_", "-")]

        # Never leave I/O blank — dual-write has no prompt text, so summarize.
        obs_input = _truncate(input) if input is not None else {
            "agent": agent,
            "model": model or None,
            "provider": provider,
            "source": "usage_dual_write",
        }
        if output is not None:
            obs_output = _truncate(output)
        elif status != "success":
            obs_output = {
                "status": status,
                "error": (str(error_message)[:500] if error_message else None),
            }
        else:
            obs_output = {
                "status": status,
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
                "image_count": int(image_count or 0),
            }
            if latency_ms is not None:
                obs_output["latency_ms"] = latency_ms

        prop_kwargs: dict[str, Any] = {"tags": feature_tags}
        uid = _user_str(user_id)
        if uid:
            prop_kwargs["user_id"] = uid
        if guest_session_id:
            prop_kwargs["session_id"] = str(guest_session_id)

        def _emit() -> None:
            obs = client.start_observation(
                name=obs_name,
                as_type="generation",
                model=model or None,
                input=obs_input,
                output=obs_output,
                metadata=meta,
                usage_details=usage_details or None,
                cost_details=cost_details,
                level="ERROR" if status != "success" else "DEFAULT",
                status_message=(str(error_message)[:500] if error_message else None),
            )
            obs.end()

        try:
            from langfuse import propagate_attributes

            with propagate_attributes(**prop_kwargs):
                _emit()
        except Exception:
            _emit()
    except Exception:
        logger.debug("Langfuse record_generation failed", exc_info=True)


def flush() -> None:
    """Flush pending Langfuse events. Fail-open."""
    client = _client
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        logger.debug("Langfuse flush failed", exc_info=True)


def shutdown() -> None:
    """Flush and shut down the Langfuse client. Fail-open."""
    global _client, _init_attempted
    client = _client
    if client is None:
        return
    try:
        client.shutdown()
    except Exception:
        try:
            client.flush()
        except Exception:
            logger.debug("Langfuse shutdown/flush failed", exc_info=True)
    finally:
        _client = None
        _init_attempted = False


def reset_for_tests() -> None:
    """Clear cached client state (unit tests only)."""
    global _client, _init_attempted
    _client = None
    _init_attempted = False
    try:
        _suppress_dual_write.set(False)
    except Exception:
        pass
