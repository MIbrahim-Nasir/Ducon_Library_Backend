import json
import os
from pathlib import Path

from dotenv import load_dotenv

import app.chromadb as chromadb
from app.gemini_embedder import GeminiEmbeddingModel, infer_mime_type
from app.ml import get_active_embedding_profile, get_collection_names

load_dotenv()

DESCRIPTIONS_DATA = os.getenv("DESCRIPTIONS_DATA")
PROJECTS_DATA = os.getenv("PROJECTS_DATA", "data/ducon_library_projects.json")
IMAGES_FOLDER = os.getenv("IMAGES_FOLDER")


def _build_filename(item: dict) -> str:
    if item.get("filename"):
        return str(item["filename"])
    return f"{item['project']}-part{item['part']}.png"


def _to_json_text(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _build_fused_text(image_item: dict, project_item: dict) -> str:
    return "\n\n".join(
        [
            "IMAGE_ITEM_JSON:",
            _to_json_text(image_item),
            "PROJECT_ITEM_JSON:",
            _to_json_text(project_item),
        ]
    )


def ingest_fused_embeddings():
    if not DESCRIPTIONS_DATA:
        raise ValueError("DESCRIPTIONS_DATA is not set in .env")
    if not IMAGES_FOLDER:
        raise ValueError("IMAGES_FOLDER is not set in .env")

    with open(DESCRIPTIONS_DATA, "r", encoding="utf-8") as file:
        image_items: list[dict] = json.load(file)

    with open(PROJECTS_DATA, "r", encoding="utf-8") as file:
        project_items: list[dict] = json.load(file)

    projects_map: dict[str, dict] = {
        item.get("project", ""): item for item in project_items if item.get("project")
    }

    collection_names = get_collection_names()
    fused_collection = chromadb.get_db_collection(collection_names["fused"])
    embedder = GeminiEmbeddingModel()

    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []

    images_folder_path = Path(IMAGES_FOLDER)
    skipped = 0

    for image_item in image_items:
        filename = _build_filename(image_item)
        project_key = image_item.get("project", "")
        project_item = projects_map.get(project_key, {})
        image_path = images_folder_path / filename

        if not image_path.exists():
            print(f"Skipping missing image: {filename}")
            skipped += 1
            continue

        fused_text = _build_fused_text(image_item=image_item, project_item=project_item)

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        mime_type = infer_mime_type(filename)
        vector = embedder.get_image_and_text_embedding(
            text=fused_text,
            image_bytes=image_bytes,
            mime_type=mime_type,
        )

        ids.append(filename)
        documents.append(fused_text)
        embeddings.append(vector)
        metadatas.append(
            {
                "filename": filename,
                "project": project_key,
                "part": image_item.get("part"),
                "variant": "gemini_fused",
                "embedding_model": embedder.model_name,
                "embedding_profile": get_active_embedding_profile(),
            }
        )

        print(f"Embedded fused item: {filename}")

    if not ids:
        raise RuntimeError("No embeddings were generated")

    fused_collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Success! Fused embeddings upserted: {len(ids)} | skipped: {skipped}")
    print(f"Collection: {collection_names['fused']}")


if __name__ == "__main__":
    ingest_fused_embeddings()
