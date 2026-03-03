import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.db.models import Generation, User
from app.db.schema import GenerationResponse
from app.auth import get_current_user

router = APIRouter(prefix="/generations", tags=["generations"])

# Base directory where all generation images are stored
OUTPUTS_DIR = Path("outputs")


def _generation_path(user_id: int, filename: str) -> Path:
    """Resolve the file path for a user's generation image."""
    return OUTPUTS_DIR / str(user_id) / filename


@router.get("", response_model=list[GenerationResponse])
async def list_generations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Generation).where(Generation.user_id == current_user.id)
        .order_by(Generation.generated_at.desc())
    )
    return result.scalars().all()


@router.get("/{generation_id}/image")
async def get_generation_image(
    generation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Serve a private generation image. Only the owning user can access it."""
    result = await db.execute(
        select(Generation).where(
            Generation.id == generation_id,
            Generation.user_id == current_user.id,
        )
    )
    generation = result.scalar_one_or_none()
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")

    # generation.url stores the relative filename, e.g. "someimage_uuid.png"
    file_path = _generation_path(current_user.id, generation.generation_name)
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

    # Delete file from disk
    file_path = _generation_path(current_user.id, generation.generation_name)
    if file_path.exists():
        os.remove(file_path)

    await db.delete(generation)
    await db.commit()
