import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, RedirectResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.db.models import Generation, User
from app.db.schema import GenerationResponse
from app.auth import get_current_user
from app import storage

router = APIRouter(prefix="/generations", tags=["generations"])
logger = logging.getLogger(__name__)


def _to_response(gen: Generation) -> GenerationResponse:
    """Build a GenerationResponse with fresh signed_url / before_url injected."""
    data = GenerationResponse.model_validate(gen)
    data.signed_url = storage.get_generation_url(gen.id, gen.url)
    if gen.source_image_url:
        try:
            data.before_url = storage.get_generation_source_url(gen.id, gen.source_image_url)
        except Exception:
            data.before_url = None
    return data


@router.get("", response_model=list[GenerationResponse])
async def list_generations(
    limit: Optional[int] = Query(None, ge=1, le=500, description="Max rows to return (omit for all)."),
    offset: int = Query(0, ge=0, description="Rows to skip (for pagination)."),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Pagination is opt-in: omitting `limit` preserves the original
    # return-everything behaviour so existing clients keep working.
    stmt = (
        select(Generation)
        .where(Generation.user_id == current_user.id)
        .order_by(Generation.generated_at.desc())
    )
    if limit is not None:
        stmt = stmt.offset(offset).limit(limit)
    elif offset:
        stmt = stmt.offset(offset)

    result = await db.execute(stmt)
    return [_to_response(g) for g in result.scalars().all()]


@router.get("/{generation_id}/before-image")
async def get_generation_before_image(
    generation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Serve the persisted user-space (before) photo for a generation."""
    result = await db.execute(
        select(Generation).where(
            Generation.id == generation_id,
            Generation.user_id == current_user.id,
        )
    )
    generation = result.scalar_one_or_none()
    if not generation or not generation.source_image_url:
        raise HTTPException(status_code=404, detail="Before image not found")

    if storage.CLOUD_STORAGE:
        if storage.should_proxy_generation_images():
            image_bytes = await asyncio.to_thread(
                storage.read_generation_bytes, generation.source_image_url
            )
            if not image_bytes:
                raise HTTPException(status_code=404, detail="Before image file not found")
            return Response(
                content=image_bytes,
                media_type="image/png",
                headers=storage.image_response_headers(),
            )
        signed_url = storage.get_generation_source_url(generation.id, generation.source_image_url)
        return RedirectResponse(
            url=signed_url,
            status_code=302,
            headers=storage.image_response_headers(),
        )

    file_path = storage.serve_generation_source_path(generation.source_image_url)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Before image file not found on server")
    return FileResponse(
        file_path,
        media_type="image/png",
        headers=storage.image_response_headers(),
    )


@router.get("/{generation_id}/image")
async def get_generation_image(
    generation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Cloud mode: 302 redirect to a fresh presigned R2 URL.
    Local mode: serve the file directly via FileResponse.
    """
    result = await db.execute(
        select(Generation).where(
            Generation.id == generation_id,
            Generation.user_id == current_user.id,
        )
    )
    generation = result.scalar_one_or_none()
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")

    if storage.CLOUD_STORAGE:
        if storage.should_proxy_generation_images():
            image_bytes = await asyncio.to_thread(
                storage.read_generation_bytes, generation.url
            )
            if not image_bytes:
                raise HTTPException(
                    status_code=404,
                    detail="Image file not found in cloud storage",
                )
            logger.info(
                "Serving generation %s image via proxy (%d bytes)",
                generation_id,
                len(image_bytes),
            )
            return Response(
                content=image_bytes,
                media_type="image/png",
                headers=storage.image_response_headers(),
            )

        try:
            signed_url = storage.get_generation_url(generation.id, generation.url)
        except Exception as exc:
            logger.exception(
                "Failed to presign generation %s image (key=%r)",
                generation_id,
                generation.url,
            )
            raise HTTPException(
                status_code=503,
                detail="Unable to generate image download URL",
            ) from exc
        if not signed_url:
            raise HTTPException(
                status_code=503,
                detail="Unable to generate image download URL",
            )
        logger.info("Serving generation %s image via redirect", generation_id)
        return RedirectResponse(
            url=signed_url,
            status_code=302,
            headers=storage.image_response_headers(),
        )

    file_path = storage.serve_local_path(generation.url)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on server")
    return FileResponse(
        file_path,
        media_type="image/png",
        headers=storage.image_response_headers(),
    )


@router.delete("/{generation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_generation(
    generation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Generation).where(
            Generation.id == generation_id,
            Generation.user_id == current_user.id,
        )
    )
    generation = result.scalar_one_or_none()
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")

    storage.delete_generation(generation.url)

    await db.delete(generation)
    await db.commit()
