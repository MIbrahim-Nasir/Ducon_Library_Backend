"""
ingest_images.py
────────────────
Syncs metadata.json → images DB table, then writes the DB-assigned id
back into metadata.json as the "id" field on each entry.

Run once after adding new images:
    uv run python ingest_images.py

Safe to re-run — uses upsert by filename (new rows inserted, existing rows updated).
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:ducondb@localhost/Ducon_Library")
METADATA_PATH = Path(os.getenv("METADATA_PATH", "data/images/metadata.json"))

engine = create_async_engine(DATABASE_URL, echo=False)
session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def sync():
    # Import here to avoid circular issues when running standalone
    from app.db.models import Image

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    async with session_maker() as db:
        for item in items:
            filename = item["filename"]
            url = item.get("url", "")
            name = item.get("name")
            theme = item.get("theme")

            # Check if this image already exists by filename
            result = await db.execute(select(Image).where(Image.filename == filename))
            image = result.scalar_one_or_none()

            if image:
                # Update fields in case they changed in metadata
                image.name = name
                image.url = url
                image.theme = theme
            else:
                image = Image(filename=filename, url=url, name=name, theme=theme)
                db.add(image)

            # Flush to get the DB-assigned id before moving to next item
            await db.flush()
            item["id"] = image.id

        await db.commit()

    # Write ids back into metadata.json
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Done. Synced {len(items)} images. IDs written back to {METADATA_PATH}")


if __name__ == "__main__":
    asyncio.run(sync())
