from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
from contextlib import asynccontextmanager
from . import chromadb
from app.ml import TextEmbeddingModel, MultimodalEmbeddingModel
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from . import gemini
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
    app.state.text_collection = chromadb.get_db_collection("text_store")
    app.state.image_collection = chromadb.get_db_collection("image_store")

    app.state.text_model = TextEmbeddingModel()
    app.state.image_model = MultimodalEmbeddingModel()

    yield

app = FastAPI(lifespan=lifespan)

app.include_router(auth_router)
app.include_router(bookmarks_router)
app.include_router(generations_router)
app.include_router(images_router)

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

def search_by_text(app, query: str, limit: int):
    embedding = app.state.text_model.get_embedding(query)
    results = chromadb.retrieve(app.state.text_collection, embedding, limit)
    return results["ids"]

def search_by_image_text(app, query: str, limit: int):
    embedding = app.state.image_model.get_text_embedding(query)
    results = chromadb.retrieve(app.state.image_collection, embedding, limit)
    return results["ids"]

async def search_by_image(app, image_file: UploadFile, limit: int):
    image_data = image_file.file.read()
    image_obj = Image.open(io.BytesIO(image_data))
    embedding = app.state.image_model.get_image_embedding(image_obj)
    results = chromadb.retrieve(app.state.image_collection, embedding, limit)
    return results["ids"]

@app.post("/search")
async def search(query: Optional[str]= Form(None), file: Optional[UploadFile] = File(None), limit: int = 5):
    results = {}
    if query:
        text_result = search_by_text(app, query=query, limit=limit)
        if text_result:
            results["text_result"] = text_result
        image_text_result = search_by_image_text(app, query=query, limit=limit)
        if image_text_result:
            results["image_text_result"] = image_text_result

    if file:
        image_result = await search_by_image(app, image_file=file, limit=limit)
        if image_result:
            results["image_result"] = image_result

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

    gemini.combine_images(
        generation_filename,
        ducon_image,
        user_img,
        prompt=prompt,
        subfolder=user_subfolder,
    )

    generation_url = f"/generations/{current_user.id}/{generation_filename}"
    db_generation = Generation(
        user_id=current_user.id,
        generation_name=generation_filename,
        url=generation_url,
        ducon_image_id=ducon_db_image.id,
    )
    db.add(db_generation)
    await db.flush()  # populate db_generation.id from DB
    await db.commit()

    return {
        "id": db_generation.id,
        "generation_name": generation_filename,
        "url": generation_url,
        "ducon_image_id": ducon_db_image.id,
    }


def main():
    print("Hello from ducon-library-backend!")


if __name__ == "__main__":
    main()
