"""
ingest_images.py
────────────────
Syncs metadata.json → images DB table.

Default mode
    Constructs each image URL from IMAGES_BASE_URL + filename, upserts the DB,
    then writes both the DB-assigned id AND the constructed url back into
    metadata.json.

    uv run python ingest_images.py

--sync mode
    Reads the url already present in each metadata.json entry (e.g. an R2 public
    URL written by upload_images_to_r2.py), upserts the DB using that url, then
    writes back only the DB-assigned id — the url in metadata.json is left as-is.

    uv run python ingest_images.py --sync

Safe to re-run in either mode — upserts by filename.

Required env vars (.env):
    DATABASE_URL      postgresql+asyncpg://...
    IMAGES_BASE_URL   https://your-vps.com/images   (no trailing slash, default mode only)
    METADATA_PATH     data/images/metadata.json      (default)
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

load_dotenv()

DATABASE_URL    = os.getenv("DATABASE_URL",     "postgresql+asyncpg://postgres:ducondb@localhost/Ducon_Library")
IMAGES_BASE_URL = os.getenv("IMAGES_BASE_URL",  "http://localhost:8000/data/images").rstrip("/")
METADATA_PATH   = Path(os.getenv("METADATA_PATH", "data/images/metadata.json"))

engine        = create_async_engine(DATABASE_URL, echo=False)
session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def run(sync_mode: bool):
    from app.db.models import Image

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    inserted = 0
    updated  = 0

    async with session_maker() as db:
        for item in items:
            filename = item["filename"]
            name     = item.get("name")

            if sync_mode:
                # Use the url already stored in metadata (e.g. R2 public URL)
                url = item.get("url")
                if not url:
                    print(f"Skipping '{filename}' — no url found in metadata entry")
                    continue
            else:
                # Construct url from base url + filename
                url = f"{IMAGES_BASE_URL}/{filename}"

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

            await db.flush()
            item["id"] = image.id

            if not sync_mode:
                # Also write back the constructed url in default mode
                item["url"] = url

        await db.commit()

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    written = "id" if sync_mode else "id+url"
    print(f"Done. {inserted} inserted, {updated} updated. {written} written to {METADATA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync metadata.json to the images DB table.")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Read urls from metadata.json as-is instead of constructing from IMAGES_BASE_URL.",
    )
    args = parser.parse_args()

    asyncio.run(run(sync_mode=args.sync))
