from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
from contextlib import asynccontextmanager
from . import chromadb
from app.ml import GeminiEmbeddingModel
import io
import os
import httpx
from PIL import Image
import pillow_heif
from fastapi.middleware.cors import CORSMiddleware

# Register HEIC/HEIF support so Pillow can open iPhone images
pillow_heif.register_heif_opener()
from fastapi.staticfiles import StaticFiles
from typing import Optional
from . import gemini
from . import storage
from pathlib import Path
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from dotenv import load_dotenv

load_dotenv()

from app.routers.auth import router as auth_router
from app.routers.bookmarks import router as bookmarks_router
from app.routers.generations import router as generations_router
from app.routers.images import router as images_router
from app.routers.guest import (
    router as guest_router,
    get_or_create_guest_session,
    increment_guest_count,
    log_guest_consent,
    verify_turnstile,
)
from app.db.database import get_db
from app.db.models import Generation, GuestGeneration, Image as DBImage
from app.auth import get_current_user, get_optional_user
from datetime import datetime, timedelta, timezone


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.collection = chromadb.get_db_collection()
    app.state.embedding_model = GeminiEmbeddingModel()
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(auth_router)
app.include_router(bookmarks_router)
app.include_router(generations_router)
app.include_router(images_router)
app.include_router(guest_router)

# Serve public library images statically
app.mount("/public/images", StaticFiles(directory="data/images"), name="public_images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"], # Allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)



# @app.get("/")
# def read_root(request: Request):
#     col = request.app.state.collection

#     return {"item_count": request.app.state.collection.count() }

def _mime_type(filename: str) -> str:
    lower = (filename or "").lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


@app.post("/search")
async def search(
    query: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    limit: int = 5,
):
    model: GeminiEmbeddingModel = app.state.embedding_model
    collection = app.state.collection

    if query and file:
        image_bytes = await file.read()
        embedding = model.get_multimodal_embedding(query, image_bytes, _mime_type(file.filename))
    elif query:
        embedding = model.get_text_embedding(query)
    elif file:
        image_bytes = await file.read()
        embedding = model.get_image_embedding(image_bytes, _mime_type(file.filename))
    else:
        return {"results": []}

    results = chromadb.retrieve(collection, embedding, limit)
    return {"results": results["ids"]}

MAX_IMAGE_PX = 2048


import json as _json
import time as _time

_IMAGE_INFO_JSON_PATH = os.getenv("IMAGE_INFO_JSON_PATH", "")
_IMAGE_INFO_CACHE: list[dict] | None = None
_IMAGE_INFO_CACHE_AT: float = 0.0
_IMAGE_INFO_CACHE_TTL = 60 * 60 * 24  # 1 day


def _load_image_info() -> list[dict]:
    global _IMAGE_INFO_CACHE, _IMAGE_INFO_CACHE_AT
    if _IMAGE_INFO_CACHE is not None and (_time.time() - _IMAGE_INFO_CACHE_AT) < _IMAGE_INFO_CACHE_TTL:
        return _IMAGE_INFO_CACHE
    if not _IMAGE_INFO_JSON_PATH:
        return []
    if _IMAGE_INFO_JSON_PATH.startswith("http://") or _IMAGE_INFO_JSON_PATH.startswith("https://"):
        import httpx as _httpx
        r = _httpx.get(_IMAGE_INFO_JSON_PATH, timeout=30)
        if r.status_code != 200:
            return []
        data = r.json()
    else:
        p = Path(_IMAGE_INFO_JSON_PATH)
        if not p.exists():
            return []
        with open(p, "r", encoding="utf-8") as f:
            data = _json.load(f)
    _IMAGE_INFO_CACHE = data
    _IMAGE_INFO_CACHE_AT = _time.time()
    return _IMAGE_INFO_CACHE


def _get_image_metadata(filename: str) -> dict | None:
    """Look up a Ducon image's info entry by filename. Returns None if not found."""
    try:
        return next((item for item in _load_image_info() if item.get("filename") == filename), None)
    except Exception:
        return None


async def _load_ducon_image(db_image) -> Image.Image:
    """
    Loads a Ducon catalogue image as a PIL Image.
    Tries the local data/images/ folder first; falls back to fetching
    from the R2 URL stored in the database if the file isn't present locally.
    """
    local_path = Path(__file__).parent.parent / "data" / "images" / db_image.filename
    if local_path.exists():
        return Image.open(local_path)

    if not db_image.url:
        raise HTTPException(
            status_code=404,
            detail=f"Image '{db_image.filename}' not found locally and has no remote URL."
        )

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(db_image.url)
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch image '{db_image.filename}' from remote URL (HTTP {response.status_code})."
        )
    return Image.open(io.BytesIO(response.content))


def _normalize_user_image(image_bytes: bytes) -> Image.Image:
    """
    Normalizes a user-uploaded image:
    - Converts HEIC/HEIF (iPhone) to JPEG-compatible RGB
    - Downsizes to MAX_IMAGE_PX on the longest side if larger, preserving aspect ratio
    - Always returns an RGB PIL Image
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    w, h = img.size
    longest = max(w, h)
    if longest > MAX_IMAGE_PX:
        scale = MAX_IMAGE_PX / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return img


@app.post("/autogenerate-images")
async def auto_generate_images(
    request: Request,
    image_id: int = Form(...),
    user_image: Optional[UploadFile] = File(None),
    second_image_id: Optional[int] = Form(None),
    prompt: str = Form(None),
    enhance_prompt: bool = Form(True),
    use_image_info: bool = Form(True),
    cf_turnstile_token: Optional[str] = Form(None),
    current_user=Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    # ── Auth gate ─────────────────────────────────────────────────────────────
    # Accept authenticated users (Bearer token) OR guests (X-Guest-Session-Id).
    is_guest = current_user is None
    if is_guest:
        raw_session_id = request.headers.get("x-guest-session-id")
        if not raw_session_id:
            raise HTTPException(
                status_code=400,
                detail="Authentication required. Provide a Bearer token or X-Guest-Session-Id header.",
            )
        try:
            uuid.UUID(raw_session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid X-Guest-Session-Id. Must be a valid UUID.")

        client_ip = request.client.host if request.client else "unknown"

        if not await verify_turnstile(cf_turnstile_token or "", client_ip):
            raise HTTPException(
                status_code=403,
                detail="Bot verification failed. Please try again.",
            )

        guest_session = await get_or_create_guest_session(db, raw_session_id, client_ip)

    # ── Image validation ──────────────────────────────────────────────────────
    if user_image is None and second_image_id is None:
        raise HTTPException(status_code=422, detail="Provide either 'user_image' (file upload) or 'second_image_id' (Ducon image id).")
    if user_image is not None and second_image_id is not None:
        raise HTTPException(status_code=422, detail="Provide only one of 'user_image' or 'second_image_id', not both.")

    # ── Resolve primary Ducon image ───────────────────────────────────────────
    result = await db.execute(select(DBImage).where(DBImage.id == image_id))
    ducon_db_image = result.scalar_one_or_none()
    if not ducon_db_image:
        raise HTTPException(status_code=404, detail="Image not found")

    ducon_image = await _load_ducon_image(ducon_db_image)

    # ── Resolve second image ──────────────────────────────────────────────────
    second_image_name: str | None = None
    second_db_image = None
    if user_image is not None:
        image_data = await user_image.read()
        user_img = _normalize_user_image(image_data)
    else:
        result2 = await db.execute(select(DBImage).where(DBImage.id == second_image_id))
        second_db_image = result2.scalar_one_or_none()
        if not second_db_image:
            raise HTTPException(status_code=404, detail=f"Second image with id {second_image_id} not found")
        user_img = await _load_ducon_image(second_db_image)
        second_image_name = second_db_image.name or second_db_image.filename

    # ── Step 1: Build Nano Banana prompt ─────────────────────────────────────
    # enhance_prompt=True (default): Flash-Lite always runs; user prompt is a hint.
    # enhance_prompt=False: skip Flash-Lite, send user prompt directly to Nano Banana.
    if enhance_prompt:
        prompt = gemini.generate_prompt(
            image1=ducon_image,
            image1_name=ducon_db_image.name or ducon_db_image.filename,
            image1_metadata=_get_image_metadata(ducon_db_image.filename) if use_image_info else None,
            image2=user_img,
            image2_name=second_image_name,
            image2_metadata=_get_image_metadata(second_db_image.filename) if (second_db_image and use_image_info) else None,
            user_hint=prompt or None,
        )
    else:
        if not prompt:
            raise HTTPException(
                status_code=422,
                detail="A prompt is required when enhance_prompt is disabled.",
            )

    # ── Step 2: Generate image ────────────────────────────────────────────────
    generation_filename = f"{ducon_db_image.filename}_{uuid.uuid4()}.png"

    if is_guest:
        subfolder = f"guests/{raw_session_id}"
    else:
        subfolder = str(current_user.id)

    gemini.combine_images(
        generation_filename,
        ducon_image,
        user_img,
        prompt=prompt,
        subfolder=subfolder,
    )

    # ── Step 2b: Evaluate and optionally regenerate ───────────────────────────
    # Only runs when enhance_prompt=True (Flash-Lite wrote the prompt).
    # The generated image is on local disk at this point — before any R2 upload.
    # If evaluation rejects it, combine_images() overwrites the local file with a
    # corrected generation. Only the final image (approved or retried) is uploaded.
    if enhance_prompt:
        local_output = Path("outputs") / subfolder / generation_filename
        try:
            generated_img = Image.open(local_output)
            approved, revised_prompt = gemini.evaluate_generation(
                image1=ducon_image,
                image2=user_img,
                generated=generated_img,
                prompt_used=prompt,
                image1_name=ducon_db_image.name or ducon_db_image.filename,
                image2_name=second_image_name,
            )
            if not approved and revised_prompt:
                print(f"[Evaluator] Regenerating with revised prompt...")
                gemini.combine_images(
                    generation_filename,
                    ducon_image,
                    user_img,
                    prompt=revised_prompt,
                    subfolder=subfolder,
                )
        except Exception as e:
            # Evaluation failure must not block the response — proceed with original
            print(f"[Evaluator] Skipped due to error: {e}")

    # ── Step 3: Persist & respond ─────────────────────────────────────────────
    if is_guest:
        stored_key = storage.save_guest_generation(raw_session_id, generation_filename)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=48)

        db_generation = GuestGeneration(
            guest_session_id=raw_session_id,
            generation_name=generation_filename,
            url=stored_key,
            ducon_image_id=ducon_db_image.id,
            expires_at=expires_at,
        )
        db.add(db_generation)
        await increment_guest_count(db, guest_session)
        await log_guest_consent(db, raw_session_id, client_ip)
        await db.flush()
        await db.commit()

        signed_url = storage.get_guest_generation_url(db_generation.id, stored_key)
        return {
            "id": db_generation.id,
            "generation_name": generation_filename,
            "signed_url": signed_url,
            "ducon_image_id": ducon_db_image.id,
            "expires_at": expires_at.isoformat(),
        }
    else:
        stored_key = storage.save_generation(current_user.id, generation_filename)

        db_generation = Generation(
            user_id=current_user.id,
            generation_name=generation_filename,
            url=stored_key,
            ducon_image_id=ducon_db_image.id,
        )
        db.add(db_generation)
        await db.flush()
        await db.commit()

        signed_url = storage.get_generation_url(db_generation.id, stored_key)
        return {
            "id": db_generation.id,
            "generation_name": generation_filename,
            "url": stored_key,
            "signed_url": signed_url,
            "ducon_image_id": ducon_db_image.id,
        }


def main():
    print("Hello from ducon-library-backend!")


if __name__ == "__main__":
    main()
