import app.chromadb as chromadb
from app.ml import (
    MultimodalEmbeddingModel,
    TextEmbeddingModel,
    get_active_embedding_profile,
    get_collection_names,
)
import argparse
import os
from pathlib import Path
from PIL import Image
import json
from dotenv import load_dotenv

load_dotenv()

DESCRIPTIONS_DATA = os.getenv("DESCRIPTIONS_DATA")
PROJECTS_DATA = os.getenv("PROJECTS_DATA", "data/ducon_library_projects.json")

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

with open(PROJECTS_DATA, "r", encoding="utf-8") as file:
    PROJECTS = json.load(file)

PROJECTS_MAP: dict[str, dict] = {
    item.get("project", ""): item for item in PROJECTS if item.get("project")
}

collection_names = get_collection_names()
image_collection = chromadb.get_db_collection(collection_names["image"])
text_collection = chromadb.get_db_collection(collection_names["text"])

clip_model = MultimodalEmbeddingModel()
text_model = TextEmbeddingModel()

print(f"Embedding profile: {get_active_embedding_profile()}")
print(f"Using collections: text={collection_names['text']} image={collection_names['image']}")

def _build_filename(item: dict) -> str:
    return f"{item['project']}-part{item['part']}.png"


def _join_sentence_parts(parts: list[str]) -> str:
    cleaned = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
    return ". ".join(cleaned)


def _build_style_text(item: dict, project_info: dict) -> str:
    project_theme = project_info.get("theme", "")
    project_tags = project_info.get("tags", "")
    # image_name = item.get("name", "")
    # description = item.get("description", "")
    # return _join_sentence_parts([project_theme, image_name, description])
    return _join_sentence_parts([project_theme, project_tags])


def _build_features_text(item: dict, project_info: dict) -> str:
    features_in_image = item.get("features_in_image") or []
    # project_tags = project_info.get("tags") or []

    features_text = ", ".join(str(x).strip() for x in features_in_image if str(x).strip())
    # project_tags_text = ", ".join(str(x).strip() for x in project_tags if str(x).strip())
    # return _join_sentence_parts([features_text, project_tags_text])
    return _join_sentence_parts([features_text])


def ingest_text(include_overall: bool, include_style: bool, include_features: bool):
    ids = []
    documents = []
    embeddings = []
    metadatas = []

    if not (include_overall or include_style or include_features):
        raise ValueError("At least one text embedding variant must be enabled")

    variant_counts = {"overall": 0, "style": 0, "features": 0}

    for item in DATA:
        filename = _build_filename(item)
        project_info = PROJECTS_MAP.get(item.get("project", ""), {})

        if include_overall:
            overall_text = item.get("overall_description") or ""
            if overall_text.strip():
                ids.append(f"{filename}::overall")
                documents.append(overall_text)
                embeddings.append(text_model.get_document_embedding(overall_text))
                metadatas.append({
                    "filename": filename,
                    "project": item.get("project", ""),
                    "variant": "overall",
                })
                variant_counts["overall"] += 1

        if include_style:
            style_text = _build_style_text(item, project_info)
            if style_text.strip():
                ids.append(f"{filename}::style")
                documents.append(style_text)
                embeddings.append(text_model.get_document_embedding(style_text))
                metadatas.append({
                    "filename": filename,
                    "project": item.get("project", ""),
                    "variant": "style",
                })
                variant_counts["style"] += 1

        if include_features:
            features_text = _build_features_text(item, project_info)
            if features_text.strip():
                ids.append(f"{filename}::features")
                documents.append(features_text)
                embeddings.append(text_model.get_document_embedding(features_text))
                metadatas.append({
                    "filename": filename,
                    "project": item.get("project", ""),
                    "variant": "features",
                })
                variant_counts["features"] += 1

    print(f"Inserting {len(ids)} items into ChromaDB...")
    print(
        "Text variants counts:",
        f"overall={variant_counts['overall']},",
        f"style={variant_counts['style']},",
        f"features={variant_counts['features']}",
    )

    text_collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print("Success! Text Data Ingested")

def ingest_images(img_folder: str):
    ids = []
    embeddings = []
    metadatas = []

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
                continue

            ids.append(file.name)
            embeddings.append(image_embedding)
            metadatas.append({
                "filename": file.name,
                "variant": "image",
            })

            print(f"Embedded Image {file.name}")

    image_collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest image/text embeddings into ChromaDB")
    parser.add_argument("--include-overall", action="store_true", help="Include overall_description embeddings")
    parser.add_argument("--include-style", action="store_true", help="Include style_text embeddings")
    parser.add_argument("--include-features", action="store_true", help="Include features_text embeddings")
    parser.add_argument("--skip-images", action="store_true", help="Skip image embedding ingestion")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    IMAGES_FOLDER = os.getenv("IMAGES_FOLDER")

    if not IMAGES_FOLDER:
        raise ValueError("IMAGES_FOLDER is not set in .env")

    include_overall = args.include_overall
    include_style = args.include_style
    include_features = args.include_features

    if not (include_overall or include_style or include_features):
        include_overall = True
        include_style = True
        include_features = True

    if not args.skip_images:
        ingest_images(IMAGES_FOLDER)

    ingest_text(
        include_overall=include_overall,
        include_style=include_style,
        include_features=include_features,
    )