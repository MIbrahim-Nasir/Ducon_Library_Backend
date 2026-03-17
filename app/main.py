import asyncio
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
from contextlib import asynccontextmanager
from . import chromadb
from app.ml import get_collection_names
from app.gemini_embedder import GeminiEmbeddingModel, infer_mime_type
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional, Any
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


gemini_embedder: GeminiEmbeddingModel | None = None


def get_gemini_embedder() -> GeminiEmbeddingModel:
    global gemini_embedder
    if gemini_embedder is None:
        gemini_embedder = GeminiEmbeddingModel()
    return gemini_embedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    collection_names = get_collection_names()
    app.state.fused_collection = chromadb.get_db_collection(collection_names["fused"])

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

async def search_by_text(app, query: str, limit: int):
    embedder = get_gemini_embedder()
    embedding = await asyncio.to_thread(embedder.get_text_embedding, query)
    retrieval_limit = max(limit * 3, limit)
    results = await asyncio.to_thread(chromadb.retrieve, app.state.fused_collection, embedding, retrieval_limit)
    ids, distances, variants = _collapse_result_ids_with_variants(
        ids=results.get("ids", [[]])[0],
        distances=results.get("distances", [[]])[0],
        metadatas=results.get("metadatas", [[]])[0],
        limit=limit,
    )
    return ids, distances, variants

async def search_by_image_text(app, query: str, limit: int):
    embedder = get_gemini_embedder()
    embedding = await asyncio.to_thread(embedder.get_text_embedding, query)
    retrieval_limit = max(limit * 3, limit)
    results = await asyncio.to_thread(chromadb.retrieve, app.state.fused_collection, embedding, retrieval_limit)
    ids, distances, variants = _collapse_result_ids_with_variants(
        ids=results.get("ids", [[]])[0],
        distances=results.get("distances", [[]])[0],
        metadatas=results.get("metadatas", [[]])[0],
        limit=limit,
    )
    return ids, distances, variants

async def search_by_image(app, image_file: UploadFile, limit: int):
    image_data = await image_file.read()
    embedder = get_gemini_embedder()
    mime_type = image_file.content_type or infer_mime_type(image_file.filename)
    embedding = await asyncio.to_thread(
        embedder.get_image_embedding,
        image_data,
        mime_type,
    )
    retrieval_limit = max(limit * 3, limit)
    results = await asyncio.to_thread(chromadb.retrieve, app.state.fused_collection, embedding, retrieval_limit)
    ids, distances, variants = _collapse_result_ids_with_variants(
        ids=results.get("ids", [[]])[0],
        distances=results.get("distances", [[]])[0],
        metadatas=results.get("metadatas", [[]])[0],
        limit=limit,
    )
    return ids, distances, variants


def _to_filename_id(result_id: str) -> str:
    return str(result_id).split("::", 1)[0]


def _extract_variant(result_id: str, metadata: dict | None) -> str | None:
    if isinstance(metadata, dict):
        metadata_variant = metadata.get("variant")
        if metadata_variant:
            return str(metadata_variant)

    if "::" in str(result_id):
        return str(result_id).split("::", 1)[1]

    return None


def _collapse_result_ids_with_variants(
    ids: list[str],
    distances: list[float],
    metadatas: list[dict] | None,
    limit: int,
) -> tuple[list[str], list[float], dict[str, str]]:
    best_by_filename: dict[str, dict[str, Any]] = {}

    for index, (raw_id, raw_distance) in enumerate(zip(ids, distances)):
        filename_id = _to_filename_id(raw_id)
        distance = float(raw_distance)

        metadata = None
        if isinstance(metadatas, list) and index < len(metadatas) and isinstance(metadatas[index], dict):
            metadata = metadatas[index]

        variant = _extract_variant(raw_id, metadata)

        if filename_id not in best_by_filename or distance < float(best_by_filename[filename_id]["distance"]):
            payload = {
                "distance": distance,
                "variant": variant,
            }
            best_by_filename[filename_id] = payload

    ranked = sorted(best_by_filename.items(), key=lambda item: float(item[1]["distance"]))
    if limit > 0:
        ranked = ranked[:limit]

    collapsed_ids = [item[0] for item in ranked]
    collapsed_distances = [float(item[1]["distance"]) for item in ranked]
    variants_by_id = {
        item[0]: str(item[1]["variant"])
        for item in ranked
        if item[1]["variant"] is not None
    }
    return collapsed_ids, collapsed_distances, variants_by_id


def _collapse_result_ids(ids: list[str], distances: list[float], limit: int) -> tuple[list[str], list[float]]:
    collapsed_ids, collapsed_distances, _ = _collapse_result_ids_with_variants(
        ids=ids,
        distances=distances,
        metadatas=None,
        limit=limit,
    )
    return collapsed_ids, collapsed_distances


def build_result_metrics(
    ids: list[str],
    distances: list[float],
    variants_by_id: Optional[dict[str, str]] = None,
) -> dict[str, dict[str, float | str]]:
    metrics: dict[str, dict[str, float | str]] = {}
    for result_id, distance in zip(ids, distances):
        score = 1.0 - float(distance)
        metrics[result_id] = {
            "distance": float(distance),
            "score": score,
        }
        if variants_by_id and result_id in variants_by_id:
            metrics[result_id]["variant"] = variants_by_id[result_id]
    return metrics

@app.post("/search")
async def search(query: Optional[str]= Form(None), file: Optional[UploadFile] = File(None), limit: int = 5):
    results = {}
    if query:
        text_result, text_distances, text_variants = await search_by_text(app, query=query, limit=limit)
        image_text_result, image_text_distances, image_text_variants = text_result, text_distances, text_variants
        if text_result:
            results["text_result"] = text_result
            results["text_result_metrics"] = build_result_metrics(text_result, text_distances, text_variants)
        if image_text_result:
            results["image_text_result"] = image_text_result
            results["image_text_result_metrics"] = build_result_metrics(image_text_result, image_text_distances, image_text_variants)

    if file:
        image_result, image_distances, image_variants = await search_by_image(app, image_file=file, limit=limit)
        if image_result:
            results["image_result"] = image_result
            results["image_result_metrics"] = build_result_metrics(image_result, image_distances, image_variants)

    return results

@app.post("/autogenerate-images")
async def auto_generate_images(
    image_id: int = Form(...),
    user_image: UploadFile = File(...),
    prompt: str = Form(None),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Resolve ducon image from DB
    result = await db.execute(select(DBImage).where(DBImage.id == image_id))
    ducon_db_image = result.scalar_one_or_none()
    if not ducon_db_image:
        raise HTTPException(status_code=404, detail="Image not found")

    ducon_image_path = Path(__file__).parent.parent / "data" / "images" / ducon_db_image.filename
    if not ducon_image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file '{ducon_db_image.filename}' not found on server")

    if prompt is None:
        prompt = """
                the image 1 is our company project design image of outdoor living area. take features such as the flooring, pavers, outdoor furniture, planters etc. take those features only and do not take non-outdoor features such as buildings, cars, walls, etc which are not our work. and then apply those feautures to the image 2 of the user's image in a logical and aesthetic manner. do not change the fixed features like buildings and existing structures.
                """

    ducon_image = Image.open(ducon_image_path)
    image_data = await user_image.read()
    user_img = Image.open(io.BytesIO(image_data))

    generation_filename = f"{ducon_db_image.filename}_{uuid.uuid4()}.png"
    user_subfolder = str(current_user.id)

    # Gemini generates the image to outputs/{user_id}/{filename}
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
