from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.db.models import Generation, User
from app.db.schema import GenerationResponse
from app.auth import get_current_user
from app import storage

router = APIRouter(prefix="/generations", tags=["generations"])


def _to_response(gen: Generation) -> GenerationResponse:
    """Build a GenerationResponse with a fresh signed_url injected."""
    data = GenerationResponse.model_validate(gen)
    data.signed_url = storage.get_generation_url(gen.id, gen.url)
    return data


@router.get("", response_model=list[GenerationResponse])
async def list_generations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Generation)
        .where(Generation.user_id == current_user.id)
        .order_by(Generation.generated_at.desc())
    )
    return [_to_response(g) for g in result.scalars().all()]


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
        signed_url = storage.get_generation_url(generation.id, generation.url)
        return RedirectResponse(url=signed_url, status_code=302)
    else:
        file_path = storage.serve_local_path(generation.url)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found on server")
        return FileResponse(file_path, media_type="image/png")


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
