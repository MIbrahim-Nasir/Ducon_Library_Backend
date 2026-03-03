"""
ingest_images.py
────────────────
Syncs metadata.json → images DB table, then writes the DB-assigned id
and constructed url back into metadata.json on each entry.

Run once after adding new images to the VPS and updating metadata.json:
    uv run python ingest_images.py

Safe to re-run — upserts by filename (new rows inserted, existing rows updated).

Required env vars (.env):
    DATABASE_URL      postgresql+asyncpg://...
    IMAGES_BASE_URL   https://your-vps.com/images   (no trailing slash)
    METADATA_PATH     data/images/metadata.json      (default)
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

load_dotenv()

DATABASE_URL    = os.getenv("DATABASE_URL",     "postgresql+asyncpg://postgres:ducondb@localhost/Ducon_Library")
IMAGES_BASE_URL = os.getenv("IMAGES_BASE_URL",  "http://localhost:8000/public/images").rstrip("/")
METADATA_PATH   = Path(os.getenv("METADATA_PATH", "data/images/metadata.json"))

engine        = create_async_engine(DATABASE_URL, echo=False)
session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def sync():
    from app.db.models import Image

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    inserted = 0
    updated  = 0

    async with session_maker() as db:
        for item in items:
            filename = item["filename"]
            url      = f"{IMAGES_BASE_URL}/{filename}"
            name     = item.get("name")

            result = await db.execute(select(Image).where(Image.filename == filename))
            image  = result.scalar_one_or_none()

            if image:
                image.name = name
                image.url  = url
                updated += 1
            else:
                image = Image(filename=filename, url=url, name=name)
                db.add(image)
                inserted += 1

            await db.flush()         # get DB-assigned id immediately
            item["id"]  = image.id
            item["url"] = url        # write url into metadata.json too

        await db.commit()

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Done. {inserted} inserted, {updated} updated. id+url written to {METADATA_PATH}")


if __name__ == "__main__":
    asyncio.run(sync())
