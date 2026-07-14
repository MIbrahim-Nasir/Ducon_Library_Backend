"""
app/tool_generate_image.py
==========================
Multi-image generation for the Ducon Designer Chat Agent.

Core idea (from Gemini docs research)
──────────────────────────────────────
Gemini image-generation models have no filename registry.  You reference
images by their *position* ("image 1", "image 2", …) **and** by an
interleaved text label that appears directly before each inline image in
the parts array.  Putting the task prompt **last** is the recommended
pattern because the model sees all images before reading its instructions,
which significantly improves instruction-following accuracy.

  parts = [
      "user space (image 1):",
      <PIL Image – user space>,
      "Ducon pergola (image 2):",
      <PIL Image – pergola>,
      "Apply the pergola (image 2) into the space (image 1). Photorealistic.",
  ]

Model limits (max total ≤ 14 across all image slots)
──────────────────────────────────────────────────────
  pro   (gemini-3-pro-image-preview)       6 obj + 5 char = up to 11 practical
  flash (gemini-3.1-flash-image-preview)  10 obj + 4 char = up to 14 practical

We enforce a safe cap of MAX_IMAGES = 10 to stay within both models.

Source resolution
──────────────────
Each image descriptor carries a `source` string that the endpoint resolves:

  type "catalog_id"    source = "42"            → DB lookup by id
  type "catalog_name"  source = "pool_coping"   → DB lookup by name/filename
  type "generation_id" source = "123"            → DB lookup in generations table
  type "url"           source = "https://…"      → HTTP fetch
  type "file"          (no source)               → UploadFile already in memory
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Literal, Optional

import httpx
from PIL import Image
from google.genai.types import GenerateContentConfig, ImageConfig, Modality, ThinkingConfig
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Generation, Image as DBImage
from app import storage
from app.gemini import (
    get_gemini_client,
    verify_prompt,
    evaluate_generation,
    evaluate_generation_multi,
    PROMPT_VERIFY_MAX_ROUNDS,
    GEN_EVAL_MAX_ROUNDS,
    build_quality_notice,
)
from app.image_gen_agent import ImageGenAgent
from app.image_utils import normalize_user_image
from app.prompt_generator_session import DesignerPromptSession

from typing import TYPE_CHECKING, Awaitable, Callable
if TYPE_CHECKING:
    from app.benchmark.types import BenchmarkStep, StepMetrics

logger = logging.getLogger(__name__)

# ── Model config ──────────────────────────────────────────────────────────────

from app.admin.settings_store import cfg

MULTI_IMAGE_PRO_MODEL = "gemini-3-pro-image-preview"
MULTI_IMAGE_FLASH_MODEL = "gemini-3.1-flash-image-preview"

def _multi_image_pro_model() -> str:
    return cfg("MULTI_IMAGE_PRO_MODEL", MULTI_IMAGE_PRO_MODEL)


def _multi_image_flash_model() -> str:
    return cfg("MULTI_IMAGE_FLASH_MODEL", MULTI_IMAGE_FLASH_MODEL)


def _model_map() -> dict:
    return {
        "pro": _multi_image_pro_model(),
        "flash": _multi_image_flash_model(),
    }

_NANO_BANANA_2 = "gemini-3.1-flash-image-preview"

MULTI_IMAGE_THINKING_LEVEL = "High"
_LIVE_DEBUG: bool = False


def _multi_image_thinking() -> str:
    return cfg("MULTI_IMAGE_THINKING_LEVEL", MULTI_IMAGE_THINKING_LEVEL)


def _dbg(*args) -> None:
    if cfg("LIVE_DEBUG", _LIVE_DEBUG):
        try:
            print(*args)
        except (UnicodeEncodeError, OSError):
            # Windows consoles default to cp1252; a debug print must never
            # crash the generation pipeline over a symbol like '▶'.
            pass


def _response_usage(response) -> dict | None:
    usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
    if usage is None:
        return None
    return usage.model_dump(exclude_none=True) if hasattr(usage, "model_dump") else usage

# Safe upper bound — keep well inside both models' hard limits.
MAX_IMAGES = 10


def _is_user_space_label(label: str) -> bool:
    normalized = (label or "").strip().lower()
    if any(token in normalized for token in (
        "user space",
        "client space",
        "client photo",
        "user photo",
        "uploaded space",
        "original space",
        "space photo",
        "user upload",
        "upload",
        "attached space",
        "my space",
        "site photo",
        "room photo",
    )):
        return True
    # Voice/chat shorthand: "user terrace", "client garden", "my pool photo"
    for prefix in ("user ", "client ", "my ", "our "):
        if normalized.startswith(prefix):
            return True
    return False


def _is_design_direction_label(label: str) -> bool:
    normalized = (label or "").strip().lower()
    if any(token in normalized for token in (
        "design direction",
        "direction design",
        "ducon design direction",
        "ducon design",
        "reference design",
        "catalog reference",
        "design reference",
    )):
        return True
    # Soft signal: "Ducon …" catalog refs that are not the user space.
    if normalized.startswith("ducon ") and not _is_user_space_label(normalized):
        return True
    return False


def classify_image_roles(labels: list[str]) -> dict[str, int | list[int]] | None:
    """
    Map labels to user space, design direction, and product reference indices.
    Studio order: [user space, design direction, product, product, ...]
    """
    if len(labels) < 2:
        return None

    design_idx = next(
        (i for i, label in enumerate(labels) if _is_design_direction_label(label)),
        None,
    )
    user_idx = next(
        (
            i for i, label in enumerate(labels)
            if _is_user_space_label(label) and not _is_design_direction_label(label)
        ),
        None,
    )

    if user_idx is None:
        user_idx = len(labels) - 1

    if design_idx is None:
        design_idx = next((i for i in range(len(labels)) if i != user_idx), 0)

    product_idxs = [
        i for i in range(len(labels))
        if i != user_idx and i != design_idx
    ]

    return {
        "user_space_index": user_idx,
        "design_direction_index": design_idx,
        "product_indices": product_idxs,
    }


SUPPORTED_ASPECT_RATIOS = frozenset({"1:1", "4:3", "3:4", "16:9", "9:16"})
ASPECT_RATIO_AUTO = "auto"


def infer_aspect_ratio_from_image(img: Image.Image) -> Optional[str]:
    """Snap image dimensions to the nearest supported generation ratio."""
    width, height = img.size
    if not width or not height:
        return None
    target = width / height
    candidates = {"1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "16:9": 16 / 9, "9:16": 9 / 16}
    return min(candidates, key=lambda ratio: abs(candidates[ratio] - target))


def resolve_aspect_ratio(
    aspect_ratio: Optional[str],
    *,
    reference_image: Optional[Image.Image] = None,
) -> Optional[str]:
    """
    Normalize aspect_ratio for Gemini ImageConfig only.

    ``"auto"`` is a Gemini ImageConfig value (match input images). Do not pass it
    to OpenAI/FAL/benchmark providers that expect fixed ratios — those paths must
    map or omit ``auto`` themselves.

    - ``"auto"`` (or empty when a reference image is present) → Gemini ``auto``
      (preferred for designer jobs).
    - Explicit ``1:1`` / ``16:9`` / … → passed through unchanged.
    - Unknown values → infer from ``reference_image`` when available, else ``auto``.
    - ``None`` with no reference → ``None`` (no ImageConfig lock; legacy behaviour).
    """
    raw = (aspect_ratio or "").strip().lower()
    if raw in ("auto", "match", "match photo", "match the photo"):
        return ASPECT_RATIO_AUTO
    if raw in SUPPORTED_ASPECT_RATIOS:
        return raw
    if raw:
        token = raw.split()[0]
        if token in SUPPORTED_ASPECT_RATIOS:
            return token
        if reference_image is not None:
            inferred = infer_aspect_ratio_from_image(reference_image)
            if inferred:
                logger.info(
                    "[MultiImageGen] unknown aspect_ratio=%r, inferred %s from reference",
                    aspect_ratio,
                    inferred,
                )
                return inferred
        return ASPECT_RATIO_AUTO
    if reference_image is not None:
        return ASPECT_RATIO_AUTO
    return None


def _select_qc_roles(labels: list[str]) -> tuple[int, int] | None:
    """
    Return (reference_index, user_space_index) for post-generation QC.

    Designer jobs pass images as [client space, Ducon refs...]. Older direct
    multi-image calls often pass [Ducon ref, user space]. The evaluator needs
    explicit roles, not positional guesses.
    """
    roles = classify_image_roles(labels)
    if roles is None:
        return None
    return roles["design_direction_index"], roles["user_space_index"]


def _should_enhance_direct_prompt(prompt_session: Optional[DesignerPromptSession], labels: list[str]) -> bool:
    """Direct chat/voice/studio generations should use the unified prompt generator path."""
    if prompt_session is not None:
        return False
    return classify_image_roles(labels) is not None

# ── Image descriptor ──────────────────────────────────────────────────────────

SourceType = Literal["catalog_id", "catalog_name", "generation_id", "url", "file"]


@dataclass
class ImageDescriptor:
    label:      str
    type:       SourceType
    source:     Optional[str] = None   # None only for type=="file"
    pil_image:  Optional[Image.Image] = None   # pre-resolved file uploads


# ── Source resolution ─────────────────────────────────────────────────────────

async def _assert_public_url(url: str) -> None:
    """SSRF guard: only allow http(s) URLs that resolve to public IP addresses.

    Blocks loopback, private, link-local and other reserved ranges (e.g. cloud
    metadata endpoints) so an attacker-supplied `url` source can't be used to
    reach internal services.
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError("URL has no host.")

    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, parsed.port or 0)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host: {host}") from exc

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError("URL resolves to a non-public address and is not allowed.")


async def _fetch_url(url: str) -> Image.Image:
    await _assert_public_url(url)
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        resp = await client.get(url)
    if resp.status_code != 200:
        raise ValueError(f"Failed to fetch image from URL (HTTP {resp.status_code}): {url}")
    # PIL decode is CPU-bound — offload so the event loop keeps serving requests.
    content = resp.content
    return await asyncio.to_thread(lambda: Image.open(io.BytesIO(content)).convert("RGB"))


async def _load_generation_image(row, *, label: str) -> Image.Image:
    """Load a Generation or GuestGeneration row's image bytes as RGB PIL."""
    if row.url.startswith("http://") or row.url.startswith("https://"):
        return await _fetch_url(row.url)
    # Guest generations use a different signed-URL helper than user generations.
    from app.db.models import GuestGeneration

    if isinstance(row, GuestGeneration):
        signed = storage.get_guest_generation_url(row.id, row.url)
    else:
        signed = storage.get_generation_url(row.id, row.url)
    if signed.startswith("/"):
        local = storage.serve_local_path(row.url)
        if not local.exists():
            raise ValueError(f"Image '{label}': local generation file missing.")
        return await asyncio.to_thread(lambda: Image.open(local).convert("RGB"))
    return await _fetch_url(signed)


async def _resolve_descriptor(
    desc: ImageDescriptor,
    db: AsyncSession,
    *,
    user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None,
) -> Image.Image:
    """Resolve one ImageDescriptor to a PIL Image.

    ``user_id`` / ``guest_session_id`` scope generation_id lookups so a caller
    cannot pull another account's generation into their own composition.
    """
    if desc.type == "file":
        if desc.pil_image is None:
            raise ValueError(f"Image '{desc.label}': type=file but no pil_image provided.")
        return desc.pil_image

    if desc.type == "url":
        return await _fetch_url(desc.source)

    if desc.type in ("catalog_id", "catalog_name"):
        q = select(DBImage)
        if desc.type == "catalog_id":
            try:
                q = q.where(DBImage.id == int(desc.source))
            except (ValueError, TypeError):
                raise ValueError(f"Image '{desc.label}': invalid catalog ID '{desc.source}'.")
        else:
            q = q.where(
                (DBImage.name == desc.source) | (DBImage.filename == desc.source)
            )
        row = (await db.execute(q)).scalar_one_or_none()
        if not row:
            raise ValueError(f"Image '{desc.label}': catalog entry not found for '{desc.source}'.")
        # Load from local disk or remote URL
        from app.image_utils import load_ducon_image
        return await load_ducon_image(row)

    if desc.type == "generation_id":
        try:
            gen_id = int(desc.source)
        except (ValueError, TypeError):
            raise ValueError(f"Image '{desc.label}': invalid generation ID '{desc.source}'.")

        if guest_session_id is not None:
            from app.db.models import GuestGeneration

            guest_q = select(GuestGeneration).where(
                GuestGeneration.id == gen_id,
                GuestGeneration.guest_session_id == guest_session_id,
            )
            guest_row = (await db.execute(guest_q)).scalar_one_or_none()
            if not guest_row:
                raise ValueError(f"Image '{desc.label}': generation {gen_id} not found.")
            return await _load_generation_image(guest_row, label=desc.label)

        gen_q = select(Generation).where(Generation.id == gen_id)
        if user_id is not None:
            gen_q = gen_q.where(Generation.user_id == user_id)
        row = (await db.execute(gen_q)).scalar_one_or_none()
        if not row:
            raise ValueError(f"Image '{desc.label}': generation {gen_id} not found.")
        return await _load_generation_image(row, label=desc.label)

    raise ValueError(f"Image '{desc.label}': unknown source type '{desc.type}'.")


# ── Gemini generation (blocking) ──────────────────────────────────────────────


def _call_verify(images, labels, prompt, prompt_model, prompt_thinking, metrics):
    """Positional-arg wrapper for ``verify_prompt`` (keyword-only overrides)."""
    return verify_prompt(
        images, labels, prompt,
        prompt_model=prompt_model,
        prompt_thinking=prompt_thinking,
        metrics=metrics,
    )


def _call_eval_multi(images, labels, generated, prompt_used,
                     user_space_index, design_direction_index, product_indices,
                     prompt_model, prompt_thinking, metrics):
    """Positional-arg wrapper for ``evaluate_generation_multi``."""
    return evaluate_generation_multi(
        images, labels, generated, prompt_used,
        user_space_index=user_space_index,
        design_direction_index=design_direction_index,
        product_indices=product_indices,
        prompt_model=prompt_model,
        prompt_thinking=prompt_thinking,
        metrics=metrics,
    )


def _call_eval(image1, image2, generated, prompt_used,
               image1_name, image2_name,
               prompt_model, prompt_thinking, metrics):
    """Positional-arg wrapper for ``evaluate_generation``."""
    return evaluate_generation(
        image1, image2, generated, prompt_used, image1_name, image2_name,
        prompt_model=prompt_model,
        prompt_thinking=prompt_thinking,
        metrics=metrics,
    )


def _run_generation_sync(
    model_id: str,
    pil_images: list[Image.Image],
    labels: list[str],
    prompt: str,
    aspect_ratio: Optional[str] = None,
    user_id: Optional[int] = None,
    *,
    image_thinking: Optional[str] = None,
    metrics: "Optional[StepMetrics]" = None,
) -> bytes:
    """
    Build the interleaved-label parts array, call Gemini, return raw PNG bytes.
    Runs synchronously — must be called via asyncio.to_thread.

    Parts order (recommended pattern — prompt last):
        "<label> (image N):"  →  <PIL Image>  →  …  →  <prompt>

    ``image_thinking`` overrides the live ``MULTI_IMAGE_THINKING_LEVEL`` cfg
    value (per-call config for the benchmark runner). ``metrics`` enables
    instrumentation: usage is recorded with a benchmark agent tag and
    tokens/cost accumulated into the holder.
    """
    client = get_gemini_client()
    is_flash = (model_id == _NANO_BANANA_2)

    # Build parts: alternating label-string / PIL-image, then prompt last.
    contents: list = []
    for i, (img, label) in enumerate(zip(pil_images, labels), start=1):
        contents.append(f"{label} (image {i}):")
        contents.append(img)
    contents.append(prompt)

    _think = image_thinking or _multi_image_thinking()

    # Config
    image_cfg = ImageConfig(aspect_ratio=aspect_ratio) if aspect_ratio else None
    config = GenerateContentConfig(
        response_modalities=[Modality.TEXT, Modality.IMAGE],
        image_config=image_cfg,
        thinking_config=ThinkingConfig(
            thinking_level=_think,
            include_thoughts=False,
        ) if is_flash else None,
    )

    logger.info("[MultiImageGen] model=%s  images=%d  aspect_ratio=%s",
                model_id, len(pil_images), aspect_ratio)
    _dbg(
        "[MULTI_IMAGE ▶ GEMINI]",
        {
            "model": model_id,
            "images": [
                {"label": label, "size": img.size, "mode": img.mode}
                for label, img in zip(labels, pil_images)
            ],
            "prompt_chars": len(prompt),
            "prompt_preview": prompt[:1200],
            "config": config.model_dump(exclude_none=True) if hasattr(config, "model_dump") else repr(config),
        },
    )

    import time as _time
    t0 = _time.perf_counter()
    response = None
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config,
            )
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            retryable = any(
                token in msg
                for token in ("503", "429", "unavailable", "timeout", "temporarily", "deadline")
            )
            if attempt == 0 and retryable:
                logger.warning(
                    "[MultiImageGen] transient Gemini error (attempt 1): %s — retrying once",
                    exc,
                )
                _time.sleep(2.5)
                continue
            raise
    if response is None and last_exc is not None:
        raise last_exc
    latency_ms = int((_time.perf_counter() - t0) * 1000)
    _dbg("[MULTI_IMAGE ◀ GEMINI]", {"usage": _response_usage(response)})

    if metrics is not None:
        from app.gemini import _record_metrics
        _record_metrics(
            response,
            agent="bench_image",
            model=model_id,
            metrics=metrics,
            image_count=1,
            latency_ms=latency_ms,
            user_id=user_id,
        )
    else:
        from app.admin.usage_helpers import record_from_response
        record_from_response(response, agent="multi_image", model=model_id, user_id=user_id, image_count=1, latency_ms=latency_ms)

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise RuntimeError(
            "Gemini returned no candidates (empty or blocked response)."
        )
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        raise RuntimeError(
            "Gemini returned no image data (missing content parts)."
        )
    for part in parts:
        if part.text:
            logger.debug("[MultiImageGen] model text: %s", part.text[:200])
            _dbg("[MULTI_IMAGE ◀ TEXT]", part.text[:1200])
        elif part.inline_data:
            _dbg("[MULTI_IMAGE ◀ IMAGE]", {"bytes": len(part.inline_data.data)})
            return part.inline_data.data   # raw bytes (PNG)

    raise RuntimeError("Gemini returned no image data.")


# ── Public API ────────────────────────────────────────────────────────────────

async def generate_multi_image(
    *,
    user_id:           Optional[int] = None,
    guest_session_id:  Optional[str] = None,
    guest_session=None,
    prompt:            str,
    descriptors:       list[ImageDescriptor],
    model:             str = "pro",
    aspect_ratio:      Optional[str] = None,
    db:                Optional[AsyncSession] = None,
    output_prefix:     str = "multi",
    enable_verify:     bool = True,
    prompt_session:    Optional[DesignerPromptSession] = None,
    image_model:       Optional[str] = None,
    image_thinking:    Optional[str] = None,
    prompt_model:      Optional[str] = None,
    prompt_thinking:   Optional[str] = None,
    max_eval_rounds:   Optional[int] = None,
    max_prompt_verify_rounds: Optional[int] = None,
    dry_run:           bool = False,
    on_step:           "Optional[Callable[[BenchmarkStep], Awaitable[None]]]" = None,
) -> dict:
    """
    Generate an image from multiple input images and a prompt.

    Args:
        user_id:      Authenticated user's DB id (for storage path).
        prompt:       Full task prompt for the generation model.
        descriptors:  Ordered list of ImageDescriptor objects.
        model:        "pro" or "flash" (ignored when ``image_model`` is given).
        aspect_ratio: Optional output aspect ratio: "auto" (match inputs),
                      "16:9", "1:1", etc. Omit for no ImageConfig lock.
        db:           SQLAlchemy async session (for catalog/generation lookups).
                      May be ``None`` when ``dry_run=True`` and every descriptor
                      is a ``file`` descriptor carrying a ``pil_image``.

        image_model / image_thinking / prompt_model / prompt_thinking /
        max_eval_rounds / max_prompt_verify_rounds: per-call overrides used by
        the benchmark runner to run many configs concurrently without mutating
        global ``cfg()`` state. When ``image_model`` is supplied it is used
        directly as the model id and the ``model`` pro/flash alias is skipped.

        dry_run: when True, skip DB writes, storage saves, and ``outputs/``
        persistence. The returned dict is extended with ``pil_images`` (list of
        PIL Image) and ``metrics`` (aggregated ``StepMetrics``) plus
        ``image_bytes`` (raw PNG bytes of the final image). No artifacts land
        on disk or in the database.

        on_step: optional async callback invoked at each step boundary with a
        ``BenchmarkStep`` (a ``running`` event before the call and a
        ``completed``/``failed`` event after). Used by the benchmark runner
        for live streaming + instrumentation.

    Returns:
        {
            "id":              int,            # absent in dry_run
            "generation_name": str,
            "url":             str,            # absent in dry_run
            "signed_url":      str,            # absent in dry_run
            "model_used":      str,
            "images_used":     [str, …],
            "pil_images":      [PIL.Image, …], # dry_run only
            "image_bytes":     bytes,          # dry_run only
            "metrics":         StepMetrics,    # dry_run only (aggregated)
            "steps":           [BenchmarkStep, …],  # dry_run only
            "retries":         int,            # dry_run only
            "final_prompt":    str,            # dry_run only
        }

    Raises:
        ValueError   — bad args (invalid source, too many images, etc.)
        RuntimeError — Gemini returned no image
    """
    if not dry_run and (user_id is None) == (guest_session_id is None):
        raise ValueError("Provide exactly one of user_id or guest_session_id.")

    if not descriptors:
        raise ValueError("At least one image descriptor is required.")
    if len(descriptors) > MAX_IMAGES:
        raise ValueError(
            f"Too many images: {len(descriptors)} supplied, maximum is {MAX_IMAGES}."
        )
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    # ── Model selection ──────────────────────────────────────────────────────
    # An explicit image_model (raw model id, e.g. from a benchmark config)
    # bypasses the pro/flash alias resolution entirely.
    if image_model:
        model_id = image_model
    else:
        model_id = _model_map().get(model)
        if model_id is None:
            raise ValueError(f"Unknown model '{model}'. Use 'pro' or 'flash'.")

    verify_rounds = (
        max_prompt_verify_rounds if max_prompt_verify_rounds is not None
        else PROMPT_VERIFY_MAX_ROUNDS
    )

    # Aggregated metrics across all steps (dry_run / benchmark).
    from app.benchmark.types import StepMetrics
    agg_metrics = StepMetrics() if (dry_run or on_step is not None) else None
    step_index = 0
    steps_recorded: list = []
    retries = 0
    final_prompt_used = prompt

    async def _emit(step_kind: str, mdl: str, thinking: Optional[str],
                    status: str, started_at: float, ended_at: Optional[float],
                    step_metrics: Optional[StepMetrics], *,
                    prompt_used: Optional[str] = None,
                    error: Optional[str] = None) -> None:
        nonlocal step_index
        if on_step is None:
            return
        from app.benchmark.types import BenchmarkStep as _BenchmarkStep
        dur = int((ended_at - started_at) * 1000) if ended_at else None
        step = _BenchmarkStep(
            index=step_index,
            kind=step_kind,
            model=mdl or "",
            thinking=thinking,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=dur,
            prompt_used=prompt_used,
            tokens_in=step_metrics.tokens_in if step_metrics else None,
            tokens_out=step_metrics.tokens_out if step_metrics else None,
            image_count=step_metrics.image_count if step_metrics else None,
            cost_usd=step_metrics.cost_usd if step_metrics else None,
            error=error,
        )
        for i, existing in enumerate(steps_recorded):
            if existing.index == step.index and existing.kind == step.kind:
                steps_recorded[i] = step
                break
        else:
            steps_recorded.append(step)
        if status != "running":
            step_index += 1
        try:
            await on_step(step)
        except Exception as cb_exc:  # never let a stream callback break the run
            logger.warning("[MultiImageGen] on_step callback failed: %s", cb_exc)

    # ── Resolve all image sources ─────────────────────────────────────────────
    pil_images: list[Image.Image] = []
    labels:     list[str]         = []

    for desc in descriptors:
        img = (
            await _resolve_descriptor(
                desc,
                db,
                user_id=user_id,
                guest_session_id=guest_session_id,
            )
            if db is not None
            else desc.pil_image
        )
        if img is None:
            raise ValueError(f"Image '{desc.label}': could not resolve to a PIL image.")
        # Normalise: cap size + ensure RGB. Decode/resize/re-encode is CPU-bound,
        # so offload per image to keep the event loop responsive.
        img = await asyncio.to_thread(lambda: normalize_user_image(_pil_to_bytes(img)))
        pil_images.append(img)
        labels.append(desc.label)
        _dbg(
            "[MULTI_IMAGE ▶ SOURCE]",
            {
                "label": desc.label,
                "type": desc.type,
                "source": desc.source,
                "image": {"size": img.size, "mode": img.mode},
            },
        )

    images_used = [f"{label} (image {i})" for i, label in enumerate(labels, 1)]

    roles = classify_image_roles(labels)
    ref_idx = roles["user_space_index"] if roles else 0
    ref_for_aspect = pil_images[ref_idx] if pil_images else None
    resolved_aspect = resolve_aspect_ratio(aspect_ratio, reference_image=ref_for_aspect)

    # Capture the user's space photo bytes for authoritative before/after URLs.
    source_image_bytes: bytes | None = None
    try:
        if roles is not None and roles.get("user_space_index") is not None:
            source_image_bytes = await asyncio.to_thread(
                _pil_to_bytes, pil_images[int(roles["user_space_index"])]
            )
    except Exception as exc:
        logger.warning("[MultiImageGen] could not snapshot source image: %s", exc)
        source_image_bytes = None

    # Direct multi-image calls (chat/voice quick generation) historically looked
    # best when the backend prompt generator analyzed the Ducon reference and the
    # user space before generation. Designer jobs already use DesignerPromptSession,
    # so only enhance direct calls here.
    active_prompt = prompt
    image_roles = roles
    image_gen_agent: ImageGenAgent | None = None
    if enable_verify and _should_enhance_direct_prompt(prompt_session, labels) and image_roles:
        user_idx = image_roles["user_space_index"]
        design_idx = image_roles["design_direction_index"]
        product_idxs = image_roles["product_indices"]
        image_gen_agent = ImageGenAgent(
            image1_name=labels[design_idx],
            image2_name=labels[user_idx],
            image2_is_user_space=True,
            labels=labels,
            user_space_index=user_idx,
            design_direction_index=design_idx,
            product_indices=product_idxs,
            prompt_model=prompt_model,
            prompt_thinking=prompt_thinking,
            max_eval_rounds=max_eval_rounds,
        )
        _t0 = time.perf_counter()
        try:
            sm = StepMetrics() if agg_metrics is not None else None
            await _emit("prompt_initial", prompt_model or "", prompt_thinking,
                        "running", _t0, None, None)
            active_prompt = await image_gen_agent.generate_initial_prompt(
                image1=pil_images[design_idx],
                images=pil_images,
                user_hint=prompt,
                metrics=sm,
            )
            final_prompt_used = active_prompt
            if sm is not None and agg_metrics is not None:
                agg_metrics.add(model=sm.model, tokens_in=sm.tokens_in,
                                tokens_out=sm.tokens_out, image_count=sm.image_count,
                                cost_usd=sm.cost_usd)
            await _emit("prompt_initial", prompt_model or "", prompt_thinking,
                        "completed", _t0, time.perf_counter(), sm,
                        prompt_used=active_prompt)
        except Exception as exc:
            logger.warning("[MultiImageGen] ImageGenAgent prompt skipped: %s", exc)
            image_gen_agent = None
            await _emit("prompt_initial", prompt_model or "", prompt_thinking,
                        "failed", _t0, time.perf_counter(), None, error=str(exc))

    # Pre-generation verify only when not using the unified ImageGenAgent session.
    if enable_verify and image_gen_agent is None and len(pil_images) >= 1:
        current_prompt = active_prompt
        for vround in range(verify_rounds):
            try:
                _t0 = time.perf_counter()
                sm = StepMetrics() if agg_metrics is not None else None
                await _emit("verify", prompt_model or "", prompt_thinking,
                            "running", _t0, None, None)
                v_passed, v_issues, v_improved = await asyncio.to_thread(
                    _call_verify, pil_images, labels, current_prompt,
                    prompt_model, prompt_thinking, sm,
                )
                if sm is not None and agg_metrics is not None:
                    agg_metrics.add(model=sm.model, tokens_in=sm.tokens_in,
                                    tokens_out=sm.tokens_out, image_count=sm.image_count,
                                    cost_usd=sm.cost_usd)
                await _emit("verify", prompt_model or "", prompt_thinking,
                            "completed", _t0, time.perf_counter(), sm,
                            prompt_used=current_prompt)
                if v_passed:
                    logger.info("[MultiImageGen] Prompt verified on round %d", vround + 1)
                    break
                if v_improved:
                    logger.info(
                        "[MultiImageGen] Prompt verify round %d: %d issue(s), improving",
                        vround + 1, len(v_issues),
                    )
                    current_prompt = v_improved
                else:
                    logger.info(
                        "[MultiImageGen] Prompt verify round %d: failed, no improvement available",
                        vround + 1,
                    )
                    break
            except Exception as ve:
                logger.warning("[MultiImageGen] Prompt verify skipped (round %d): %s", vround + 1, ve)
                break
        active_prompt = current_prompt
        final_prompt_used = active_prompt

    # ── Generation + post-evaluation loop ─────────────────────────────────────
    # Generate → evaluate → regenerate with revised prompt up to max_eval_rounds.
    # All retries reuse the same generation_name so only the final image is stored.
    if dry_run:
        subfolder = "dry_run"
    elif guest_session_id:
        subfolder = f"guests/{guest_session_id}"
    else:
        subfolder = str(user_id)
    safe_prefix      = "".join(c for c in output_prefix if c.isalnum() or c in ("_", "-")) or "multi"
    generation_name  = f"{safe_prefix}_{uuid.uuid4().hex[:12]}.png"
    max_rounds       = (max_eval_rounds if max_eval_rounds is not None
                        else (GEN_EVAL_MAX_ROUNDS if enable_verify else 1))

    image_bytes: bytes | None = None
    generation_approved = False   # track whether any round was approved
    last_eval_issues: list[str] = []

    for gen_round in range(max_rounds):
        _t0 = time.perf_counter()
        gen_sm = StepMetrics() if agg_metrics is not None else None
        await _emit("image_gen", model_id, image_thinking, "running", _t0, None, None)
        try:
            image_bytes = await asyncio.to_thread(
                _run_generation_sync,
                model_id,
                pil_images,
                labels,
                active_prompt,
                resolved_aspect,
                user_id,
                image_thinking=image_thinking,
                metrics=gen_sm,
            )
            if gen_sm is not None and agg_metrics is not None:
                agg_metrics.add(model=gen_sm.model, tokens_in=gen_sm.tokens_in,
                                tokens_out=gen_sm.tokens_out, image_count=gen_sm.image_count,
                                cost_usd=gen_sm.cost_usd)
            await _emit("image_gen", model_id, image_thinking, "completed",
                        _t0, time.perf_counter(), gen_sm, prompt_used=active_prompt)
        except Exception as ge:
            await _emit("image_gen", model_id, image_thinking, "failed",
                        _t0, time.perf_counter(), gen_sm, error=str(ge))
            raise

        qc_roles = _select_qc_roles(labels)
        if not enable_verify or qc_roles is None:
            # Verification requires at least two input images to be meaningful.
            generation_approved = True
            break

        try:
            # PIL decode of the generated PNG is CPU-bound — offload it.
            _gen_bytes = image_bytes
            generated_pil = await asyncio.to_thread(
                lambda: Image.open(io.BytesIO(_gen_bytes)).convert("RGB")
            )
            ref_idx, user_idx = qc_roles

            # ── Evaluate ──────────────────────────────────────────────────────
            _et0 = time.perf_counter()
            eval_sm = StepMetrics() if agg_metrics is not None else None
            await _emit("eval", prompt_model or "", prompt_thinking, "running", _et0, None, None)
            if image_gen_agent is not None:
                # Two-phase: evaluate (analyse failure) then generate (write new
                # prompt) — both via the same stateful agent session so all prior
                # context is available to the prompt-writing turn.
                approved, issues = await image_gen_agent.evaluate_output(
                    generated=generated_pil,
                    gen_round=gen_round + 1,
                    metrics=eval_sm,
                )
                revised_prompt = None  # will be produced by generate_retry_prompt below
            elif image_roles and len(pil_images) > 2:
                approved, revised_prompt, issues = await asyncio.to_thread(
                    _call_eval_multi,
                    pil_images,
                    labels,
                    generated_pil,
                    active_prompt,
                    image_roles["user_space_index"],
                    image_roles["design_direction_index"],
                    image_roles["product_indices"],
                    prompt_model,
                    prompt_thinking,
                    eval_sm,
                )
            else:
                approved, revised_prompt, issues = await asyncio.to_thread(
                    _call_eval,
                    pil_images[ref_idx],
                    pil_images[user_idx],
                    generated_pil,
                    active_prompt,
                    labels[ref_idx],
                    labels[user_idx],
                    prompt_model,
                    prompt_thinking,
                    eval_sm,
                )
            if eval_sm is not None and agg_metrics is not None:
                agg_metrics.add(model=eval_sm.model, tokens_in=eval_sm.tokens_in,
                                tokens_out=eval_sm.tokens_out, image_count=eval_sm.image_count,
                                cost_usd=eval_sm.cost_usd)
            await _emit("eval", prompt_model or "", prompt_thinking, "completed",
                        _et0, time.perf_counter(), eval_sm)

            # ── Approved ──────────────────────────────────────────────────────
            if approved:
                generation_approved = True
                logger.info("[MultiImageGen] Generation approved on round %d", gen_round + 1)
                if image_gen_agent is not None:
                    image_gen_agent.schedule_post_success_improvement()
                break

            # ── Rejected ─────────────────────────────────────────────────────
            last_eval_issues = list(issues or [])
            if gen_round + 1 >= max_rounds:
                # All rounds exhausted — keep and return the LAST generation as a
                # best-effort result rather than failing the request. The user
                # prefers seeing the closest attempt over an error message.
                logger.warning(
                    "[MultiImageGen] Generation still rejected on final round %d — "
                    "returning last attempt as best-effort result",
                    gen_round + 1,
                )
                break

            logger.info(
                "[MultiImageGen] Generation rejected on round %d (%d issues) — retrying",
                gen_round + 1, len(issues),
            )
            _dbg("[MULTI_IMAGE ▶ EVAL ISSUES]", issues)
            retries += 1

            # ── Generate improved prompt ───────────────────────────────────
            if image_gen_agent is not None:
                _rt0 = time.perf_counter()
                retry_sm = StepMetrics() if agg_metrics is not None else None
                await _emit("prompt_retry", prompt_model or "", prompt_thinking,
                            "running", _rt0, None, None)
                next_prompt = await image_gen_agent.generate_retry_prompt(
                    gen_round=gen_round + 1,
                    issues=issues,
                    metrics=retry_sm,
                )
                if retry_sm is not None and agg_metrics is not None:
                    agg_metrics.add(model=retry_sm.model, tokens_in=retry_sm.tokens_in,
                                    tokens_out=retry_sm.tokens_out, image_count=retry_sm.image_count,
                                    cost_usd=retry_sm.cost_usd)
                await _emit("prompt_retry", prompt_model or "", prompt_thinking,
                            "completed", _rt0, time.perf_counter(), retry_sm,
                            prompt_used=next_prompt)
            elif prompt_session is not None:
                next_prompt = await prompt_session.revise_from_qc(
                    {
                        "verdict": "rejected",
                        "issues": issues,
                        "passed": False,
                    },
                    context_label=f"Inner generation QC round {gen_round + 1}",
                )
            elif revised_prompt:
                next_prompt = revised_prompt
            else:
                # No improved prompt could be produced — stop retrying and keep
                # the last generation as a best-effort result instead of erroring.
                logger.info(
                    "[MultiImageGen] Generation rejected on round %d — "
                    "no revised prompt available, returning last attempt",
                    gen_round + 1,
                )
                break

            # Verify the new prompt against the inputs (non-agent path only)
            if image_gen_agent is None:
                for vround in range(verify_rounds):
                    try:
                        v_p, _, v_imp = await asyncio.to_thread(
                            _call_verify, pil_images, labels, next_prompt,
                            prompt_model, prompt_thinking, None,
                        )
                        if v_p or not v_imp:
                            break
                        next_prompt = v_imp
                    except Exception as ve:
                        logger.warning("[MultiImageGen] Re-verify skipped (round %d): %s", vround + 1, ve)
                        break

            active_prompt = next_prompt
            final_prompt_used = active_prompt
            if prompt_session is not None:
                prompt_session.last_prompt = active_prompt

        except RuntimeError:
            raise
        except Exception as ev:
            # Evaluation or prompt-generation itself errored — log and use the
            # current image as-is (best-effort; avoids losing a good generation
            # due to QC infrastructure failure).
            logger.warning("[MultiImageGen] Post-gen evaluation failed (round %d): %s", gen_round + 1, ev)
            generation_approved = True
            if prompt_session is not None:
                raise RuntimeError(f"Post-generation QC failed: {ev}") from ev
            break

    if image_bytes is None:
        raise RuntimeError("No image bytes produced after generation loop.")

    # ── Dry-run: return inline, no persistence ─────────────────────────────────
    if dry_run:
        _fb = image_bytes
        final_pil = await asyncio.to_thread(
            lambda: Image.open(io.BytesIO(_fb)).convert("RGB")
        )
        result: dict = {
            "generation_name": generation_name,
            "model_used":      model_id,
            "images_used":     images_used,
            "image_bytes":     image_bytes,
            "pil_images":      [final_pil],
            "metrics":         agg_metrics,
            "steps":           steps_recorded,
            "retries":         retries,
            "final_prompt":    final_prompt_used,
            "approved":        generation_approved,
        }
        return result

    # ── Save to disk + storage ────────────────────────────────────────────────
    await asyncio.to_thread(_save_bytes_locally, image_bytes, subfolder, generation_name)
    _dbg("[MULTI_IMAGE ▶ SAVE LOCAL]", {"subfolder": subfolder, "filename": generation_name, "bytes": len(image_bytes)})

    if guest_session_id:
        from datetime import datetime, timedelta, timezone

        from app.db.models import GuestGeneration
        from app.routers.guest import increment_guest_count

        # DB row first so a crash mid-upload does not leave an orphan without a
        # row — and so we have an id for source-image keys. Upload then updates url.
        expires_at = datetime.now(timezone.utc) + timedelta(hours=48)
        pending_key = f"pending/guest/{guest_session_id}/{generation_name}"
        db_gen = GuestGeneration(
            guest_session_id=guest_session_id,
            generation_name=generation_name,
            url=pending_key,
            ducon_image_id=None,
            expires_at=expires_at,
        )
        db.add(db_gen)
        await db.flush()
        try:
            stored_key = await storage.asave_guest_generation(guest_session_id, generation_name)
            db_gen.url = stored_key
            _dbg("[MULTI_IMAGE ▶ SAVE STORAGE]", {"stored_key": stored_key, "guest_session_id": guest_session_id})
        except Exception:
            await db.rollback()
            raise
        before_url = None
        if source_image_bytes:
            try:
                source_key = await storage.asave_generation_source(
                    source_image_bytes,
                    generation_id=db_gen.id,
                    guest_session_id=guest_session_id,
                )
                db_gen.source_image_url = source_key
                before_url = storage.get_generation_source_url(db_gen.id, source_key)
            except Exception as src_exc:
                logger.warning("[MultiImageGen] source image save failed: %s", src_exc)
        if guest_session is not None:
            await increment_guest_count(db, guest_session)
        await db.commit()
        signed_url = storage.get_guest_generation_url(db_gen.id, stored_key)
    else:
        from app.db.models import Generation as Gen

        pending_key = f"pending/{user_id}/{generation_name}"
        db_gen = Gen(
            user_id=user_id,
            generation_name=generation_name,
            url=pending_key,
            ducon_image_id=None,
        )
        db.add(db_gen)
        await db.flush()
        try:
            stored_key = await storage.asave_generation(user_id, generation_name)
            db_gen.url = stored_key
            _dbg("[MULTI_IMAGE ▶ SAVE STORAGE]", {"stored_key": stored_key})
        except Exception:
            await db.rollback()
            raise

        before_url = None
        if source_image_bytes:
            try:
                source_key = await storage.asave_generation_source(
                    source_image_bytes,
                    generation_id=db_gen.id,
                    user_id=user_id,
                )
                db_gen.source_image_url = source_key
                before_url = storage.get_generation_source_url(db_gen.id, source_key)
            except Exception as src_exc:
                logger.warning("[MultiImageGen] source image save failed: %s", src_exc)
        await db.commit()
        signed_url = storage.get_generation_url(db_gen.id, stored_key)

    _dbg(
        "[MULTI_IMAGE ◀ RESULT]",
        {
            "id": db_gen.id,
            "generation_name": generation_name,
            "url": stored_key,
            "signed_url": signed_url,
            "before_url": before_url,
        },
    )

    result = {
        "id":              db_gen.id,
        "generation_name": generation_name,
        "url":             stored_key,
        "signed_url":      signed_url,
        "before_url":      before_url,
        "source_image_url": getattr(db_gen, "source_image_url", None),
        "model_used":      model_id,
        "images_used":     images_used,
        "approved":        generation_approved,
        "generation_ref":  f"gen:{db_gen.id}",
    }
    if guest_session_id:
        result["expires_at"] = expires_at.isoformat()

    # Non-blocking notice about the user's input photo quality (tilt/crop/suitability).
    if image_gen_agent is not None and image_gen_agent.input_quality:
        if not image_gen_agent.input_quality.get("ok"):
            result["input_quality"] = image_gen_agent.input_quality

    # Best-effort result after QC retries — surface remaining issues to the user.
    if not generation_approved and last_eval_issues:
        gen_notice = build_quality_notice(
            last_eval_issues,
            default_message=(
                "We saved the closest result, but some quality checks were not fully met. "
                "You can continue in chat to refine, try another direction, or retake your space photo."
            ),
            title_prefix="Visualization quality",
        )
        if gen_notice:
            result["generation_warnings"] = gen_notice

    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return buf.getvalue()


def _save_bytes_locally(image_bytes: bytes, subfolder: str, filename: str) -> None:
    """Write raw image bytes to outputs/{subfolder}/{filename}."""
    import os
    out_dir = os.path.join("outputs", subfolder)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, filename), "wb") as f:
        f.write(image_bytes)
