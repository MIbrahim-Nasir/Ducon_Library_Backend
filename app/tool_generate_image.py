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
    evaluate_generation_multi,
    PROMPT_VERIFY_MAX_ROUNDS,
    GEN_EVAL_MAX_ROUNDS,
    build_quality_notice,
)
from app.image_gen_agent import ImageGenAgent
from app.image_utils import normalize_user_image
from app.prompt_generator_session import DesignerPromptSession

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
        "user upload",
        "attached space",
        "my space",
        "site photo",
        "room photo",
    ))


def _is_design_direction_label(label: str) -> bool:
    normalized = (label or "").strip().lower()
    return any(token in normalized for token in (
        "design direction",
        "direction design",
        "ducon design direction",
        "ducon design",
        "reference design",
        "catalog reference",
        "design reference",
    ))


def classify_image_roles(labels: list[str]) -> dict[str, int | list[int]] | None:
    """
    Map labels to user space, design direction, and product reference indices.
    Studio order: [user space, design direction, product, product, ...]
    """
    if len(labels) < 2:
        return None

    user_idx = next((i for i, label in enumerate(labels) if _is_user_space_label(label)), None)
    design_idx = next(
        (i for i, label in enumerate(labels) if _is_design_direction_label(label)),
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
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


async def _resolve_descriptor(
    desc: ImageDescriptor,
    db: AsyncSession,
    *,
    user_id: Optional[int] = None,
) -> Image.Image:
    """Resolve one ImageDescriptor to a PIL Image.

    ``user_id`` scopes generation_id lookups to the requesting user so a caller
    cannot pull another user's generation into their own composition.
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
        gen_q = select(Generation).where(Generation.id == gen_id)
        if user_id is not None:
            gen_q = gen_q.where(Generation.user_id == user_id)
        row = (await db.execute(gen_q)).scalar_one_or_none()
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
    user_id: Optional[int] = None,
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
            thinking_level=_multi_image_thinking(),
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
    from app.admin.usage_helpers import record_from_response
    record_from_response(response, agent="multi_image", model=model_id, user_id=user_id, image_count=1)

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

    model_id = _model_map().get(model)
    if model_id is None:
        raise ValueError(f"Unknown model '{model}'. Use 'pro' or 'flash'.")

    # ── Resolve all image sources ─────────────────────────────────────────────
    pil_images: list[Image.Image] = []
    labels:     list[str]         = []

    for desc in descriptors:
        img = await _resolve_descriptor(desc, db, user_id=user_id)
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
    image_roles = classify_image_roles(labels)
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
        )
        try:
            active_prompt = await image_gen_agent.generate_initial_prompt(
                image1=pil_images[design_idx],
                images=pil_images,
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
    generation_approved = False   # track whether any round was approved
    last_eval_issues: list[str] = []

    for gen_round in range(max_rounds):
        image_bytes = await asyncio.to_thread(
            _run_generation_sync,
            model_id,
            pil_images,
            labels,
            active_prompt,
            aspect_ratio,
            user_id,
        )

        qc_roles = _select_qc_roles(labels)
        if not enable_verify or qc_roles is None:
            # Verification requires at least two input images to be meaningful.
            generation_approved = True
            break

        try:
            generated_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            ref_idx, user_idx = qc_roles

            # ── Evaluate ──────────────────────────────────────────────────────
            if image_gen_agent is not None:
                # Two-phase: evaluate (analyse failure) then generate (write new
                # prompt) — both via the same stateful agent session so all prior
                # context is available to the prompt-writing turn.
                approved, issues = await image_gen_agent.evaluate_output(
                    generated=generated_pil,
                    gen_round=gen_round + 1,
                )
                revised_prompt = None  # will be produced by generate_retry_prompt below
            elif image_roles and len(pil_images) > 2:
                approved, revised_prompt, issues = await asyncio.to_thread(
                    evaluate_generation_multi,
                    pil_images,
                    labels,
                    generated_pil,
                    active_prompt,
                    user_space_index=image_roles["user_space_index"],
                    design_direction_index=image_roles["design_direction_index"],
                    product_indices=image_roles["product_indices"],
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

            # ── Generate improved prompt ───────────────────────────────────
            if image_gen_agent is not None:
                # Dedicated prompt-writing turn: agent uses its own evaluation
                # analysis (already in conversation history) to write a better prompt.
                next_prompt = await image_gen_agent.generate_retry_prompt(
                    gen_round=gen_round + 1,
                    issues=issues,
                )
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

    result: dict = {
        "id":              db_gen.id,
        "generation_name": generation_name,
        "url":             stored_key,
        "signed_url":      signed_url,
        "model_used":      model_id,
        "images_used":     images_used,
    }

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
