import app.chromadb as chromadb
from app.ml import GeminiEmbeddingModel
import os
from pathlib import Path
import json
from dotenv import load_dotenv

load_dotenv()

DESCRIPTIONS_DATA = os.getenv("DESCRIPTIONS_DATA")
IMAGES_FOLDER = os.getenv("IMAGES_FOLDER")

with open(DESCRIPTIONS_DATA, "r", encoding="utf-8") as file:
    DATA = json.load(file)

collection = chromadb.get_db_collection()
model = GeminiEmbeddingModel()


def ingest_text():
    ids = []
    documents = []
    embeddings = []

    for item in DATA:
        text_content = item["overall_description"]
        filename = item["project"] + "-part" + str(item["part"]) + ".png"
        ids.append(filename)
        documents.append(text_content)
        embeddings.append(model.get_text_embedding(text_content))
        print(f"Embedded text for {filename}")

    print(f"Inserting {len(ids)} text items into ChromaDB...")
    collection.add(ids=ids, embeddings=embeddings, documents=documents)
    print("Success! Text data ingested.")


def ingest_images(img_folder: str):
    ids = []
    embeddings = []

    folder = Path(img_folder)

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            try:
                image_bytes = file.read_bytes()
                mime_type = "image/png" if file.suffix.lower() == ".png" else "image/jpeg"
                embedding = model.get_image_embedding(image_bytes, mime_type)
                ids.append(file.name)
                embeddings.append(embedding)
                print(f"Embedded image {file.name}")
            except Exception as e:
                print(f"Skipping {file.name}: {e}")

    print(f"Inserting {len(ids)} image items into ChromaDB...")
    collection.add(ids=ids, embeddings=embeddings)
    print("Success! Image data ingested.")


if __name__ == "__main__":
    ingest_images(IMAGES_FOLDER)
    ingest_text()
