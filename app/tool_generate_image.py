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
    PROMPT_VERIFY_MAX_ROUNDS,
    GEN_EVAL_MAX_ROUNDS,
)
from app.image_gen_agent import ImageGenAgent
from app.image_utils import normalize_user_image
from app.prompt_generator_session import DesignerPromptSession

logger = logging.getLogger(__name__)

# ── Model config ──────────────────────────────────────────────────────────────

MULTI_IMAGE_PRO_MODEL   = os.getenv("MULTI_IMAGE_PRO_MODEL",   "gemini-3-pro-image-preview")
MULTI_IMAGE_FLASH_MODEL = os.getenv("MULTI_IMAGE_FLASH_MODEL", "gemini-3.1-flash-image-preview")

_MODEL_MAP = {
    "pro":   MULTI_IMAGE_PRO_MODEL,
    "flash": MULTI_IMAGE_FLASH_MODEL,
}
_NANO_BANANA_2 = "gemini-3.1-flash-image-preview"

MULTI_IMAGE_THINKING_LEVEL = os.getenv("MULTI_IMAGE_THINKING_LEVEL", "High")
_LIVE_DEBUG: bool = os.getenv("LIVE_DEBUG", "").lower() in ("1", "true", "yes")


def _dbg(*args) -> None:
    if _LIVE_DEBUG:
        print(*args)


def _response_usage(response) -> dict | None:
    usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
    if usage is None:
        return None
    return usage.model_dump(exclude_none=True) if hasattr(usage, "model_dump") else usage

# Safe upper bound — keep well inside both models' hard limits.
MAX_IMAGES = 10


def _is_user_space_label(label: str) -> bool:
    normalized = (label or "").strip().lower()
    return any(token in normalized for token in (
        "user space",
        "client space",
        "client photo",
        "user photo",
        "uploaded space",
        "original space",
        "space photo",
    ))


def _select_qc_roles(labels: list[str]) -> tuple[int, int] | None:
    """
    Return (reference_index, user_space_index) for post-generation QC.

    Designer jobs pass images as [client space, Ducon refs...]. Older direct
    multi-image calls often pass [Ducon ref, user space]. The evaluator needs
    explicit roles, not positional guesses.
    """
    if len(labels) < 2:
        return None

    user_idx = next((i for i, label in enumerate(labels) if _is_user_space_label(label)), None)
    if user_idx is None:
        # Legacy fallback used by older direct calls.
        user_idx = len(labels) - 1

    ref_idx = next((i for i in range(len(labels)) if i != user_idx), None)
    if ref_idx is None:
        return None
    return ref_idx, user_idx


def _should_enhance_direct_prompt(prompt_session: Optional[DesignerPromptSession], labels: list[str]) -> bool:
    """Direct chat/voice generations should use the proven prompt generator path."""
    roles = _select_qc_roles(labels)
    # The legacy prompt generator writes for exactly: image 1 = Ducon reference,
    # image 2 = user space. Only use it when the direct call has that shape.
    return prompt_session is None and len(labels) == 2 and roles == (0, 1)

# ── Image descriptor ──────────────────────────────────────────────────────────

SourceType = Literal["catalog_id", "catalog_name", "generation_id", "url", "file"]


@dataclass
class ImageDescriptor:
    label:      str
    type:       SourceType
    source:     Optional[str] = None   # None only for type=="file"
    pil_image:  Optional[Image.Image] = None   # pre-resolved file uploads


# ── Source resolution ─────────────────────────────────────────────────────────

async def _fetch_url(url: str) -> Image.Image:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url)
    if resp.status_code != 200:
        raise ValueError(f"Failed to fetch image from URL (HTTP {resp.status_code}): {url}")
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


async def _resolve_descriptor(desc: ImageDescriptor, db: AsyncSession) -> Image.Image:
    """Resolve one ImageDescriptor to a PIL Image."""
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
        row = (await db.execute(
            select(Generation).where(Generation.id == gen_id)
        )).scalar_one_or_none()
        if not row:
            raise ValueError(f"Image '{desc.label}': generation {gen_id} not found.")
        if row.url.startswith("http://") or row.url.startswith("https://"):
            return await _fetch_url(row.url)
        # Derive signed URL through storage layer
        signed = storage.get_generation_url(row.id, row.url)
        if signed.startswith("/"):
            # Local mode: serve from disk
            local = storage.serve_local_path(row.url)
            if not local.exists():
                raise ValueError(f"Image '{desc.label}': local generation file missing.")
            return Image.open(local).convert("RGB")
        return await _fetch_url(signed)

    raise ValueError(f"Image '{desc.label}': unknown source type '{desc.type}'.")


# ── Gemini generation (blocking) ──────────────────────────────────────────────

def _run_generation_sync(
    model_id: str,
    pil_images: list[Image.Image],
    labels: list[str],
    prompt: str,
    aspect_ratio: Optional[str] = None,
) -> bytes:
    """
    Build the interleaved-label parts array, call Gemini, return raw PNG bytes.
    Runs synchronously — must be called via asyncio.to_thread.

    Parts order (recommended pattern — prompt last):
        "<label> (image N):"  →  <PIL Image>  →  …  →  <prompt>
    """
    client = get_gemini_client()
    is_flash = (model_id == _NANO_BANANA_2)

    # Build parts: alternating label-string / PIL-image, then prompt last.
    contents: list = []
    for i, (img, label) in enumerate(zip(pil_images, labels), start=1):
        contents.append(f"{label} (image {i}):")
        contents.append(img)
    contents.append(prompt)

    # Config
    image_cfg = ImageConfig(aspect_ratio=aspect_ratio) if aspect_ratio else None
    config = GenerateContentConfig(
        response_modalities=[Modality.TEXT, Modality.IMAGE],
        image_config=image_cfg,
        thinking_config=ThinkingConfig(
            thinking_level=MULTI_IMAGE_THINKING_LEVEL,
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

    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config=config,
    )
    _dbg("[MULTI_IMAGE ◀ GEMINI]", {"usage": _response_usage(response)})

    for part in response.candidates[0].content.parts:
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
    user_id:           int,
    prompt:            str,
    descriptors:       list[ImageDescriptor],
    model:             str = "pro",
    aspect_ratio:      Optional[str] = None,
    db:                AsyncSession,
    output_prefix:     str = "multi",
    enable_verify:     bool = True,
    prompt_session:    Optional[DesignerPromptSession] = None,
) -> dict:
    """
    Generate an image from multiple input images and a prompt.

    Args:
        user_id:      Authenticated user's DB id (for storage path).
        prompt:       Full task prompt for the generation model.
        descriptors:  Ordered list of ImageDescriptor objects.
        model:        "pro" or "flash".
        aspect_ratio: Optional output aspect ratio, e.g. "16:9", "1:1".
        db:           SQLAlchemy async session (for catalog/generation lookups).

    Returns:
        {
            "id":              int,
            "generation_name": str,
            "url":             str,
            "signed_url":      str,
            "model_used":      str,
            "images_used":     [str, …]   # "label (image N)" strings
        }

    Raises:
        ValueError   — bad args (invalid source, too many images, etc.)
        RuntimeError — Gemini returned no image
    """
    if not descriptors:
        raise ValueError("At least one image descriptor is required.")
    if len(descriptors) > MAX_IMAGES:
        raise ValueError(
            f"Too many images: {len(descriptors)} supplied, maximum is {MAX_IMAGES}."
        )
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    model_id = _MODEL_MAP.get(model)
    if model_id is None:
        raise ValueError(f"Unknown model '{model}'. Use 'pro' or 'flash'.")

    # ── Resolve all image sources ─────────────────────────────────────────────
    pil_images: list[Image.Image] = []
    labels:     list[str]         = []

    for desc in descriptors:
        img = await _resolve_descriptor(desc, db)
        # Normalise: cap size + ensure RGB
        img = normalize_user_image(_pil_to_bytes(img))
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

    # Direct multi-image calls (chat/voice quick generation) historically looked
    # best when the backend prompt generator analyzed the Ducon reference and the
    # user space before generation. Designer jobs already use DesignerPromptSession,
    # so only enhance direct calls here.
    active_prompt = prompt
    direct_roles = _select_qc_roles(labels)
    image_gen_agent: ImageGenAgent | None = None
    if enable_verify and _should_enhance_direct_prompt(prompt_session, labels) and direct_roles:
        ref_idx, user_idx = direct_roles
        image_gen_agent = ImageGenAgent(
            image1_name=labels[ref_idx],
            image2_name=None if _is_user_space_label(labels[user_idx]) else labels[user_idx],
            image2_is_user_space=_is_user_space_label(labels[user_idx]),
        )
        try:
            active_prompt = await image_gen_agent.generate_initial_prompt(
                image1=pil_images[ref_idx],
                image2=pil_images[user_idx],
                user_hint=prompt,
            )
        except Exception as exc:
            logger.warning("[MultiImageGen] ImageGenAgent prompt skipped: %s", exc)
            image_gen_agent = None

    # Pre-generation verify only when not using the unified ImageGenAgent session.
    if enable_verify and image_gen_agent is None and len(pil_images) >= 1:
        current_prompt = active_prompt
        for vround in range(PROMPT_VERIFY_MAX_ROUNDS):
            try:
                v_passed, v_issues, v_improved = await asyncio.to_thread(
                    verify_prompt, pil_images, labels, current_prompt
                )
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

    # ── Generation + post-evaluation loop ─────────────────────────────────────
    # Generate → evaluate → regenerate with revised prompt up to GEN_EVAL_MAX_ROUNDS.
    # All retries reuse the same generation_name so only the final image is stored.
    subfolder        = str(user_id)
    safe_prefix      = "".join(c for c in output_prefix if c.isalnum() or c in ("_", "-")) or "multi"
    generation_name  = f"{safe_prefix}_{uuid.uuid4().hex[:12]}.png"
    max_rounds       = GEN_EVAL_MAX_ROUNDS if enable_verify else 1

    image_bytes: bytes | None = None
    for gen_round in range(max_rounds):
        image_bytes = await asyncio.to_thread(
            _run_generation_sync,
            model_id,
            pil_images,
            labels,
            active_prompt,
            aspect_ratio,
        )

        qc_roles = _select_qc_roles(labels)
        if not enable_verify or qc_roles is None:
            # Verification requires at least two input images to be meaningful.
            break

        try:
            generated_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            ref_idx, user_idx = qc_roles
            if image_gen_agent is not None:
                approved, revised_prompt, issues = await image_gen_agent.evaluate_output(
                    generated=generated_pil,
                    gen_round=gen_round + 1,
                )
            else:
                approved, revised_prompt, issues = await asyncio.to_thread(
                    evaluate_generation,
                    pil_images[ref_idx],
                    pil_images[user_idx],
                    generated_pil,
                    active_prompt,
                    labels[ref_idx],
                    labels[user_idx],
                )
            if approved:
                logger.info("[MultiImageGen] Generation approved on round %d", gen_round + 1)
                if image_gen_agent is not None:
                    image_gen_agent.schedule_post_success_improvement()
                break
            if gen_round + 1 < max_rounds:
                logger.info(
                    "[MultiImageGen] Generation rejected on round %d (%d issues) — retrying",
                    gen_round + 1, len(issues),
                )
                _dbg("[MULTI_IMAGE ▶ EVAL ISSUES]", issues)
                if image_gen_agent is not None and revised_prompt:
                    next_prompt = revised_prompt
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
                    logger.info(
                        "[MultiImageGen] Generation rejected on round %d — "
                        "no revised prompt available, keeping output",
                        gen_round + 1,
                    )
                    break

                if image_gen_agent is None:
                    for vround in range(PROMPT_VERIFY_MAX_ROUNDS):
                        try:
                            v_p, _, v_imp = await asyncio.to_thread(
                                verify_prompt, pil_images, labels, next_prompt
                            )
                            if v_p or not v_imp:
                                break
                            next_prompt = v_imp
                        except Exception as ve:
                            logger.warning("[MultiImageGen] Re-verify skipped (round %d): %s", vround + 1, ve)
                            break
                active_prompt = next_prompt
                if prompt_session is not None:
                    prompt_session.last_prompt = active_prompt
            else:
                logger.info(
                    "[MultiImageGen] Generation rejected on round %d — max rounds reached, keeping output",
                    gen_round + 1,
                )
                break
        except Exception as ev:
            logger.warning("[MultiImageGen] Post-gen evaluation failed (round %d): %s", gen_round + 1, ev)
            if prompt_session is not None:
                raise RuntimeError(f"Post-generation QC failed: {ev}") from ev
            break

    if image_bytes is None:
        raise RuntimeError("No image bytes produced after generation loop.")

    # ── Save to disk + storage ────────────────────────────────────────────────
    _save_bytes_locally(image_bytes, subfolder, generation_name)
    _dbg("[MULTI_IMAGE ▶ SAVE LOCAL]", {"subfolder": subfolder, "filename": generation_name, "bytes": len(image_bytes)})
    stored_key = storage.save_generation(user_id, generation_name)
    _dbg("[MULTI_IMAGE ▶ SAVE STORAGE]", {"stored_key": stored_key})

    # ── Persist to DB ─────────────────────────────────────────────────────────
    from app.db.models import Generation as Gen
    db_gen = Gen(
        user_id=user_id,
        generation_name=generation_name,
        url=stored_key,
        ducon_image_id=None,
    )
    db.add(db_gen)
    await db.flush()
    await db.commit()

    signed_url = storage.get_generation_url(db_gen.id, stored_key)
    _dbg(
        "[MULTI_IMAGE ◀ RESULT]",
        {"id": db_gen.id, "generation_name": generation_name, "url": stored_key, "signed_url": signed_url},
    )

    return {
        "id":              db_gen.id,
        "generation_name": generation_name,
        "url":             stored_key,
        "signed_url":      signed_url,
        "model_used":      model_id,
        "images_used":     images_used,
    }


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
