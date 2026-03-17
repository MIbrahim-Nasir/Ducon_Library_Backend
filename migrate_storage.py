"""
migrate_storage.py
──────────────────
Migrates all stored files and database URLs between local disk and Cloudflare R2.

Usage:
    uv run python migrate_storage.py --to cloud   # local → R2
    uv run python migrate_storage.py --to local   # R2 → local

What it does
────────────
LOCAL → CLOUD
  images table   : Updates every images.url to the R2 public URL.
                   NOTE: images must already be uploaded to R2 (run upload_images_to_r2.py first).
  generations    : Uploads outputs/{user_id}/{filename} to R2 private bucket.
                   Generation.url key format stays the same — only the file moves.
  metadata.json  : Rewrites url fields to R2 public URLs → saves as metadata.json
                   (a backup copy as metadata.local.json is written first).
  bookmarks      : No URL fields — nothing to do.

CLOUD → LOCAL
  images table   : Updates every images.url back to the local FastAPI URL.
  generations    : Downloads R2 private objects back to outputs/{user_id}/{filename}.
  metadata.json  : Rewrites url fields back to local URLs → saves in place
                   (cloud version backed up as metadata.cloud.json).
  bookmarks      : No URL fields — nothing to do.

Required env vars (same as normal backend .env):
    DATABASE_URL, R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
    R2_PUBLIC_BUCKET, R2_PUBLIC_BASE_URL, R2_PRIVATE_BUCKET,
    IMAGES_BASE_URL  (local base, default http://localhost:8000/public/images)
    METADATA_PATH    (default data/images/metadata.json)
"""

import argparse
import asyncio
import functools
import json
import os
import shutil
from pathlib import Path
from urllib.parse import quote

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL        = os.getenv("DATABASE_URL",        "postgresql+asyncpg://postgres:ducondb@localhost/Ducon_Library")
R2_ENDPOINT_URL     = os.getenv("R2_ENDPOINT_URL",     "")
R2_ACCESS_KEY_ID    = os.getenv("R2_ACCESS_KEY_ID",    "")
R2_SECRET_ACCESS_KEY= os.getenv("R2_SECRET_ACCESS_KEY","")
R2_PUBLIC_BUCKET    = os.getenv("R2_PUBLIC_BUCKET",    "ducon-public")
R2_PUBLIC_BASE_URL  = os.getenv("R2_PUBLIC_BASE_URL",  "").rstrip("/")
R2_PRIVATE_BUCKET   = os.getenv("R2_PRIVATE_BUCKET",   "ducon-private")
IMAGES_BASE_URL     = os.getenv("IMAGES_BASE_URL",     "http://localhost:8000/public/images").rstrip("/")
R2_IMAGES_PREFIX    = os.getenv("R2_KEY_PREFIX",       "images").strip("/")
METADATA_PATH       = Path(os.getenv("METADATA_PATH",  "data/images/metadata.json"))
IMAGES_FOLDER       = Path(os.getenv("IMAGES_FOLDER",  "data/images"))
OUTPUTS_DIR         = Path("outputs")

engine        = create_async_engine(DATABASE_URL, echo=False)
session_maker = async_sessionmaker(engine, expire_on_commit=False)


# ── R2 helpers ────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def r2():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def r2_object_exists(bucket: str, key: str) -> bool:
    try:
        r2().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def r2_upload(bucket: str, key: str, local_path: Path, content_type: str = "image/png"):
    with open(local_path, "rb") as f:
        r2().put_object(Bucket=bucket, Key=key, Body=f, ContentType=content_type)


def r2_download(bucket: str, key: str, local_path: Path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    r2().download_file(bucket, key, str(local_path))


def r2_delete(bucket: str, key: str):
    r2().delete_object(Bucket=bucket, Key=key)


# ── URL helpers ───────────────────────────────────────────────────────────────
def local_image_url(filename: str) -> str:
    return f"{IMAGES_BASE_URL}/{filename}"


def cloud_image_url(filename: str) -> str:
    encoded = quote(filename, safe="")
    prefix  = f"{R2_IMAGES_PREFIX}/" if R2_IMAGES_PREFIX else ""
    return f"{R2_PUBLIC_BASE_URL}/{prefix}{encoded}"


def generation_key(user_id, filename: str) -> str:
    return f"generations/{user_id}/{filename}"


def local_generation_path(user_id, filename: str) -> Path:
    return OUTPUTS_DIR / str(user_id) / filename


# ── Migration: local → cloud ──────────────────────────────────────────────────
async def migrate_to_cloud():
    print("=" * 60)
    print("MIGRATING: local disk → Cloudflare R2")
    print("=" * 60)

    from app.db.models import Image, Generation

    async with session_maker() as db:

        # ── 1. Images table ──────────────────────────────────────────────
        print("\n[1/3] Updating images table URLs to R2 public URLs...")
        images_result = await db.execute(select(Image))
        images = images_result.scalars().all()

        images_updated = 0
        images_skipped = 0
        for img in images:
            new_url = cloud_image_url(img.filename)
            if img.url == new_url:
                images_skipped += 1
                continue

            # Verify the file exists in R2 before updating the DB
            r2_key = f"{R2_IMAGES_PREFIX}/{img.filename}" if R2_IMAGES_PREFIX else img.filename
            if not r2_object_exists(R2_PUBLIC_BUCKET, r2_key):
                print(f"  WARNING: {img.filename} not found in R2 public bucket — skipping DB update")
                images_skipped += 1
                continue

            img.url = new_url
            images_updated += 1
            print(f"  UPDATED  {img.filename}")

        await db.commit()
        print(f"  → {images_updated} updated, {images_skipped} already correct / skipped")

        # ── 2. Generations ───────────────────────────────────────────────
        print("\n[2/3] Uploading generation files to R2 private bucket...")
        gens_result = await db.execute(select(Generation))
        generations  = gens_result.scalars().all()

        gens_uploaded = 0
        gens_skipped  = 0
        gens_missing  = 0

        for gen in generations:
            # url is already in key format: generations/{user_id}/{filename}
            key      = gen.url
            parts    = key.split("/")  # ['generations', '{user_id}', '{filename}']
            user_id  = parts[1]
            filename = parts[2]
            local    = local_generation_path(user_id, filename)

            if r2_object_exists(R2_PRIVATE_BUCKET, key):
                print(f"  SKIP (already in R2) {key}")
                gens_skipped += 1
                continue

            if not local.exists():
                print(f"  MISSING local file: {local}")
                gens_missing += 1
                continue

            r2_upload(R2_PRIVATE_BUCKET, key, local)
            print(f"  UPLOADED {key}")
            gens_uploaded += 1

        # Generation.url key format does not change — no DB update needed
        print(f"  → {gens_uploaded} uploaded, {gens_skipped} already in R2, {gens_missing} missing locally")

    # ── 3. metadata.json ─────────────────────────────────────────────────────
    print("\n[3/3] Updating metadata.json URLs...")
    _update_metadata_urls(cloud_image_url, backup_suffix=".local.json")

    print("\n✓ Done. Set USE_CLOUD_STORAGE = true in .env to activate cloud mode.")


# ── Migration: cloud → local ──────────────────────────────────────────────────
async def migrate_to_local():
    print("=" * 60)
    print("MIGRATING: Cloudflare R2 → local disk")
    print("=" * 60)

    from app.db.models import Image, Generation

    async with session_maker() as db:

        # ── 1. Images table ──────────────────────────────────────────────
        print("\n[1/3] Updating images table URLs to local URLs...")
        images_result = await db.execute(select(Image))
        images = images_result.scalars().all()

        images_updated = 0
        images_skipped = 0
        for img in images:
            new_url = local_image_url(img.filename)
            if img.url == new_url:
                images_skipped += 1
                continue
            img.url = new_url
            images_updated += 1
            print(f"  UPDATED  {img.filename}")

        await db.commit()
        print(f"  → {images_updated} updated, {images_skipped} already correct")

        # ── 2. Generations ───────────────────────────────────────────────
        print("\n[2/3] Downloading generation files from R2 private bucket...")
        gens_result = await db.execute(select(Generation))
        generations  = gens_result.scalars().all()

        gens_downloaded = 0
        gens_skipped    = 0
        gens_missing    = 0

        for gen in generations:
            key      = gen.url  # generations/{user_id}/{filename}
            parts    = key.split("/")
            user_id  = parts[1]
            filename = parts[2]
            local    = local_generation_path(user_id, filename)

            if local.exists():
                print(f"  SKIP (already local) {key}")
                gens_skipped += 1
                continue

            if not r2_object_exists(R2_PRIVATE_BUCKET, key):
                print(f"  MISSING in R2: {key}")
                gens_missing += 1
                continue

            r2_download(R2_PRIVATE_BUCKET, key, local)
            print(f"  DOWNLOADED {key}")
            gens_downloaded += 1

        # Generation.url key format does not change — no DB update needed
        print(f"  → {gens_downloaded} downloaded, {gens_skipped} already local, {gens_missing} missing in R2")

    # ── 3. metadata.json ─────────────────────────────────────────────────────
    print("\n[3/3] Updating metadata.json URLs...")
    _update_metadata_urls(local_image_url, backup_suffix=".cloud.json")

    print("\n✓ Done. Set USE_CLOUD_STORAGE = false in .env to activate local mode.")


# ── Shared metadata helper ────────────────────────────────────────────────────
def _update_metadata_urls(url_fn, backup_suffix: str):
    """Rewrite the url field in metadata.json using url_fn(filename)."""
    if not METADATA_PATH.exists():
        print(f"  metadata.json not found at {METADATA_PATH} — skipping")
        return

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    # Write backup
    backup_path = METADATA_PATH.with_suffix(backup_suffix)
    shutil.copy2(METADATA_PATH, backup_path)
    print(f"  Backup written → {backup_path}")

    updated = 0
    for item in items:
        new_url = url_fn(item["filename"])
        if item.get("url") != new_url:
            item["url"] = new_url
            updated += 1

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"  → {updated} URLs updated in {METADATA_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Migrate storage between local disk and Cloudflare R2")
    parser.add_argument(
        "--to",
        choices=["cloud", "local"],
        required=True,
        help="Migration direction: 'cloud' (local→R2) or 'local' (R2→local)",
    )
    args = parser.parse_args()

    if args.to == "cloud":
        missing = [v for v in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_PUBLIC_BASE_URL") if not os.getenv(v)]
        if missing:
            print(f"ERROR: Missing env vars: {', '.join(missing)}")
            return
        asyncio.run(migrate_to_cloud())
    else:
        asyncio.run(migrate_to_local())


if __name__ == "__main__":
    main()
