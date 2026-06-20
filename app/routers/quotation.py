"""
POST /quotation

Accepts a Ducon catalog image (by DB id), the user's original space photo, and
the AI-generated visualisation, then returns a structured JSON breakdown of:
  • area_measurements — flexible surfaces (flooring, walls, platforms …) with
    estimated m² and confidence level.
  • fixed_items       — discrete products (pergola, furniture, countertops …)
    with unit counts.

Requires a valid Bearer token (authenticated users only).
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.auth import get_current_user
from app.db.database import get_db
from app.db.models import Image as DBImage, User
from app.image_utils import get_image_metadata, load_ducon_image, normalize_user_image
from app import gemini

router = APIRouter(prefix="/quotation", tags=["quotation"])


@router.post("")
async def create_quotation(
    ducon_image_id: int = Form(
        ...,
        description="ID of the Ducon catalog image that was used in the generation.",
    ),
    user_image: UploadFile = File(
        ...,
        description="User's original space photo (before the AI generation).",
    ),
    generation_image: UploadFile = File(
        ...,
        description="AI-generated visualisation with the Ducon design applied.",
    ),
    reference_measurements: Optional[str] = Form(
        None,
        description=(
            "Known real-world dimensions of the space, provided by the user. "
            "Free-text, e.g. 'The terrace is 8 m wide and 12 m deep. Pool is 4×8 m.' "
            "When supplied these take priority over visual perspective estimates."
        ),
    ),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Analyse a Ducon AI-generation against the original space and return a
    quantity / area measurement breakdown suitable for quotation purposes.

    The response contains two lists:

    **area_measurements** — for flexible surface treatments (paving, wall
    cladding, pool surrounds, platforms, etc.).  Each entry includes an
    estimated area in m², a confidence level, and the reasoning used.

    **fixed_items** — for discrete products with a fixed form factor (pergola,
    furniture, countertops, water features, etc.).  Each entry includes a
    quantity and unit.

    Estimates are visual approximations based on multi-image perspective
    analysis by Gemini 3.1 Pro.  They are indicative, not contractual.
    """

    # ── Resolve Ducon catalog image from DB ───────────────────────────────────
    result = await db.execute(select(DBImage).where(DBImage.id == ducon_image_id))
    ducon_db_image = result.scalar_one_or_none()
    if not ducon_db_image:
        raise HTTPException(
            status_code=404,
            detail=f"Ducon catalog image with id {ducon_image_id} not found.",
        )

    # ── Load & validate all three images ─────────────────────────────────────
    try:
        ducon_pil = await load_ducon_image(ducon_db_image)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to load Ducon catalog image: {exc}",
        ) from exc

    try:
        user_bytes = await user_image.read()
        if not user_bytes:
            raise HTTPException(status_code=422, detail="user_image file is empty.")
        user_pil = normalize_user_image(user_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read user_image: {exc}",
        ) from exc

    try:
        gen_bytes = await generation_image.read()
        if not gen_bytes:
            raise HTTPException(status_code=422, detail="generation_image file is empty.")
        from PIL import Image as PILImage
        import io
        gen_pil = PILImage.open(io.BytesIO(gen_bytes)).convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read generation_image: {exc}",
        ) from exc

    # ── Fetch Ducon metadata for richer product identification ────────────────
    metadata = get_image_metadata(ducon_db_image.filename)

    # ── Run ensemble analysis (3 concurrent passes + synthesis) ──────────────
    try:
        analysis = await gemini.analyze_quotation(
            ducon_pil,
            user_pil,
            gen_pil,
            metadata,
            reference_measurements or None,
        )
    except ValueError as exc:
        # Model returned unparseable JSON
        raise HTTPException(
            status_code=502,
            detail=f"Analysis model returned an unexpected response: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error during quotation analysis: {exc}",
        ) from exc

    return {
        "ducon_image_id": ducon_image_id,
        "ducon_image_name": ducon_db_image.name or ducon_db_image.filename,
        **analysis,
    }
