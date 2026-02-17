import app.db as db
from app.ml import MultimodalEmbeddingModel, TextEmbeddingModel
import os
from pathlib import Path
from PIL import Image
import json
from dotenv import load_dotenv

load_dotenv()

DESCRIPTIONS_DATA = os.getenv("DESCRIPTIONS_DATA")

DUMMY_DATA = [
    {
        "id": "img_001",
        "filename": "concrete_villa_01.jpg",
        "name": "The Concrete Box",
        "project": "Villa Rosso",
        "description": "A minimalist modern concrete villa with large glass windows, situated on a cliff overlooking the ocean. Brutalist style with gray tones.",
        "level": 1,
        "class": "Residential",
        "type": "Exterior",
        "theme": "Modern",
        "tags": ["concrete", "minimalist", "ocean-view", "brutalist"],
        "date": "2024-05-10",
        "link": "https://example.com/villa-rosso",
        "related": ["concrete_villa_02.jpg", "floorplan_v1.jpg"]
    },
    {
        "id": "img_002",
        "filename": "wooden_cabin_forest.jpg",
        "name": "Forest Retreat",
        "project": "Whispering Pines",
        "description": "A cozy rustic wooden cabin surrounded by a dense green forest. Warm lighting, timber cladding, and a stone chimney.",
        "level": 2,
        "class": "Hospitality",
        "type": "Exterior",
        "theme": "Rustic",
        "tags": ["wood", "forest", "cozy", "nature"],
        "date": "2023-11-20",
        "link": "https://example.com/whispering-pines",
        "related": ["interior_living.jpg"]
    },
    {
        "id": "img_003",
        "filename": "brick_office_downtown.jpg",
        "name": "Red Brick HQ",
        "project": "Tech Hub Central",
        "description": "A renovated industrial brick building converted into a modern office space. High ceilings, exposed pipes, and large arched windows.",
        "level": 5,
        "class": "Commercial",
        "type": "Interior",
        "theme": "Industrial",
        "tags": ["brick", "office", "industrial", "renovation"],
        "date": "2025-01-15",
        "link": "https://example.com/tech-hub",
        "related": ["meeting_room.jpg", "lobby_render.jpg"]
    }
]

with open(DESCRIPTIONS_DATA, "r", encoding="utf-8") as file:
    DATA = json.load(file)

image_collection = db.get_db_collection("image_store")
text_collection = db.get_db_collection("text_store")

clip_model = MultimodalEmbeddingModel()
text_model = TextEmbeddingModel()

# 1. Initialize lists


def ingest_text():
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for item in DATA:
        text_content = item["overall_description"]
        embeddings.append(text_model.get_embedding(text_content))
        filename = item["project"]+"-part"+f"{item["part"]}.png"
        ids.append(filename)
        documents.append(text_content)
        # meta = item.copy()
        # meta["tags"] = ", ".join(meta["tags"])
        # meta["related"] = ", ".join(meta["related"])
        # del meta["id"]
        # metadatas.append(meta)

    print(f"Inserting {len(ids)} items into ChromaDB...")

    text_collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents
    )

    print("Success! Text Data Ingested")

def ingest_images(img_folder: str):
    ids = []
    embeddings = []

    folder = Path(img_folder)

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            try:
                img = Image.open(file)

                if img.mode != 'RGB':
                    img = img.convert('RGB')

                image_embedding = clip_model.get_image_embedding(img)
            except Exception as e:
                print(f"Skipping {file.name}: {e}")

            ids.append(file.name)
            embeddings.append(image_embedding)

            print(f"Embedded Image {file.name}")

    image_collection.add(
        ids=ids,
        embeddings=embeddings
    )        

if __name__ == "__main__":
    load_dotenv()
    IMAGES_FOLDER = os.getenv("IMAGES_FOLDER")

    ingest_images(IMAGES_FOLDER)
    ingest_text()