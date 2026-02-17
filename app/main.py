from fastapi import FastAPI, Request, UploadFile, File, Form
from contextlib import asynccontextmanager
from . import db
from app.ml import TextEmbeddingModel, MultimodalEmbeddingModel
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from . import gemini
from pathlib import Path
from fastapi.staticfiles import StaticFiles


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.text_collection = db.get_db_collection("text_store")
    app.state.image_collection = db.get_db_collection("image_store")

    app.state.text_model = TextEmbeddingModel()
    app.state.image_model = MultimodalEmbeddingModel()

    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"], # Allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

app.mount("/output_folder", StaticFiles(directory="app/output_folder"), name="generations")


# @app.get("/")
# def read_root(request: Request):
#     col = request.app.state.collection

#     return {"item_count": request.app.state.collection.count() }

def search_by_text(app, query: str, limit: int):
    embedding = app.state.text_model.get_embedding(query)
    results = db.retrieve(app.state.text_collection, embedding, limit)
    return results["ids"]

def search_by_image_text(app, query: str, limit: int):
    embedding = app.state.image_model.get_text_embedding(query)
    results = db.retrieve(app.state.image_collection, embedding, limit)
    return results["ids"]

async def search_by_image(app, image_file: UploadFile, limit: int):
    image_data = image_file.file.read()
    image_obj = Image.open(io.BytesIO(image_data))
    embedding = app.state.image_model.get_image_embedding(image_obj)
    results = db.retrieve(app.state.image_collection, embedding, limit)
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

# @app.post("/autogenerate-images")
# async def auto_generate_images(ducon_images: list[str] = Form(None), user_images: list[UploadFile] = File(None), prompt: str=None):
#     generations = []
#     if ducon_images is None or user_images is None:
#         return {"error": "Missing required parameters"}, 400
    
#     if prompt is None:
#         prompt = """
#                 the image 1 is our company project design image of outdoor living area. take features such as the flooring, pavers, outdoor furniture, planters etc. take those features only and do not take non-outdoor features such as buildings, cars, walls, etc which are not our work. and then apply those feautures to the image 2 of the user's image in a logical and aesthetic manner. do not change the fixed features like buildings and existing structures.
#                 """

#     for filename in ducon_images:
#         ducon_image_path = Path(__file__).parent / "data" / "images" / filename   
#         ducon_image = Image.open(ducon_image_path)
#         for upload_file in user_images:
#             image_data = await upload_file.read()
#             user_image = Image.open(io.BytesIO(image_data))

#             generation = gemini.combine_images(filename, ducon_image, user_image, prompt=prompt)

#             generations.append(f"http://localhost:8000/outputs/{generation}")

#     return generations


def main():
    print("Hello from ducon-library-backend!")


if __name__ == "__main__":
    main()
