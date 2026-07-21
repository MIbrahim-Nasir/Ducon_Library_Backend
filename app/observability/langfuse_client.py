"""
Langfuse LLM observability — optional, fail-open.

Aligned with Langfuse skill / best-practices (good names, generations with
model+tokens, session_id / user_id / tags, nested observations, truncated I/O).

When ``LANGFUSE_ENABLED`` is false / unset, or keys are missing, every helper
is a no-op. Dual-write from ``app.admin.usage_recorder.record`` covers most
agent paths without scattering instrumentation.
"""
from __future__ import annotations

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
        text = str(value)
        return text if len(text) <= limit else text[:limit] + "…"
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
                client.set_current_trace_io(input=truncated_input)
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
    """Set root / current-trace output (assistant reply preview). Fail-open."""
    client = get_client()
    if client is None:
        return
    try:
        truncated = _truncate(output)
        client.set_current_trace_io(output=truncated)
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
    tags: Optional[list[str]] = None,
    suppress_dual_write: bool = True,
) -> Iterator[Any]:
    """
    Nested ``generation`` observation around an LLM call (model + tokens).

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
    try:
        try:
            cm = client.start_as_current_observation(
                as_type="generation",
                name=name,
                model=model,
                input=_truncate(input),
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
) -> None:
    """
    Fire-and-forget generation (post-hoc from usage sinks).

    Nests under the current observation when one is active (e.g. chat-response).
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
