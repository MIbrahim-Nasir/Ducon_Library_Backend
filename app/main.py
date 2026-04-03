from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
from contextlib import asynccontextmanager
from . import chromadb
from app.ml import GeminiEmbeddingModel
import io
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
from app.db.database import get_db
from app.db.models import Generation, Image as DBImage
from app.auth import get_current_user


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
    image_id: int = Form(...),
    user_image: Optional[UploadFile] = File(None),
    second_image_id: Optional[int] = Form(None),
    prompt: str = Form(None),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Exactly one of user_image or second_image_id must be provided
    if user_image is None and second_image_id is None:
        raise HTTPException(status_code=422, detail="Provide either 'user_image' (file upload) or 'second_image_id' (Ducon image id).")
    if user_image is not None and second_image_id is not None:
        raise HTTPException(status_code=422, detail="Provide only one of 'user_image' or 'second_image_id', not both.")

    # Resolve primary ducon image from DB
    result = await db.execute(select(DBImage).where(DBImage.id == image_id))
    ducon_db_image = result.scalar_one_or_none()
    if not ducon_db_image:
        raise HTTPException(status_code=404, detail="Image not found")

    ducon_image_path = Path(__file__).parent.parent / "data" / "images" / ducon_db_image.filename
    if not ducon_image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file '{ducon_db_image.filename}' not found on server")

    ducon_image = Image.open(ducon_image_path)

    # Resolve the second image — either an upload or another Ducon catalog image
    second_image_name: str | None = None
    if user_image is not None:
        image_data = await user_image.read()
        user_img = _normalize_user_image(image_data)
    else:
        result2 = await db.execute(select(DBImage).where(DBImage.id == second_image_id))
        second_db_image = result2.scalar_one_or_none()
        if not second_db_image:
            raise HTTPException(status_code=404, detail=f"Second image with id {second_image_id} not found")
        second_image_path = Path(__file__).parent.parent / "data" / "images" / second_db_image.filename
        if not second_image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image file '{second_db_image.filename}' not found on server")
        user_img = Image.open(second_image_path)
        second_image_name = second_db_image.name or second_db_image.filename

    # Step 1 — generate an optimal Nano Banana Pro prompt via gemini-3.1-flash-lite-preview
    # If the caller provides an explicit prompt, use it as-is (override / testing).
    if not prompt:
        prompt = gemini.generate_prompt(
            image1=ducon_image,
            image1_name=ducon_db_image.name or ducon_db_image.filename,
            image2=user_img,
            image2_name=second_image_name,  # None when image2 is a user upload
        )
        print(prompt)

    generation_filename = f"{ducon_db_image.filename}_{uuid.uuid4()}.png"
    user_subfolder = str(current_user.id)

    # Step 2 — generate the image using Nano Banana Pro (gemini-3-pro-image-preview)
    gemini.combine_images(
        generation_filename,
        ducon_image,
        user_img,
        prompt=prompt,
        subfolder=user_subfolder,
    )

    # Save to cloud (R2) or keep on local disk — storage module decides
    stored_key = storage.save_generation(current_user.id, generation_filename)

    # Persist to DB
    db_generation = Generation(
        user_id=current_user.id,
        generation_name=generation_filename,
        url=stored_key,
        ducon_image_id=ducon_db_image.id,
    )
    db.add(db_generation)
    await db.flush()
    await db.commit()

    # get_generation_url returns presigned URL (cloud) or API endpoint (local)
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
