import app.db as db
from app.ml import EmbeddingModel

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

collection = db.get_db_collection()
model = EmbeddingModel()

# 1. Initialize lists
ids = []
documents = []
metadatas = []
embeddings = []

for item in DUMMY_DATA:
    text_content = item["description"]
    embeddings.append(model.get_embedding(text_content))
    ids.append(item["id"])
    documents.append(text_content)
    meta = item.copy()
    meta["tags"] = ", ".join(meta["tags"])
    meta["related"] = ", ".join(meta["related"])
    del meta["id"]
    metadatas.append(meta)

print(f"Inserting {len(ids)} items into ChromaDB...")

collection.add(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas,
    documents=documents
)

print("Success! Data Ingested")