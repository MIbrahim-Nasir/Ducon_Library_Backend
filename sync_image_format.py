"""
sync_image_format.py
─────────────────────
After converting image files to a new format locally, this script:
  1. Uploads the new-format files to Cloudflare R2
  2. Updates filenames and URLs in metadata.json (in-place)
  3. Updates filenames and URLs in the database
  4. Renames IDs in ChromaDB (delete old + re-add with new filename ID)

Skips items whose filename already has the target extension.

  --to jpg          Target extension (jpg or png)
  --replace         Also delete the old-format objects from R2

Usage:
  uv run python sync_image_format.py --to jpg
  uv run python sync_image_format.py --to jpg --replace
  uv run python sync_image_format.py --to png --replace

Required env vars (.env):
  DATABASE_URL
  METADATA_PATH           (default: data/images/metadata.json)
  IMAGES_FOLDER           (default: data/images)
  R2_ENDPOINT_URL
  R2_ACCESS_KEY_ID
  R2_SECRET_ACCESS_KEY
  R2_PUBLIC_BUCKET
  R2_PUBLIC_BASE_URL
  R2_KEY_PREFIX           (default: images)
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import boto3
import chromadb as chromadb_lib
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
METADATA_PATH        = Path(os.getenv("METADATA_PATH",  "data/images/metadata.json"))
IMAGES_FOLDER        = Path(os.getenv("IMAGES_FOLDER",  "data/images"))
DATABASE_URL         = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:ducondb@localhost/Ducon_Library")

R2_ENDPOINT_URL      = os.getenv("R2_ENDPOINT_URL",      "")
R2_ACCESS_KEY_ID     = os.getenv("R2_ACCESS_KEY_ID",     "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_PUBLIC_BUCKET     = os.getenv("R2_PUBLIC_BUCKET",     "ducon-public")
R2_PUBLIC_BASE_URL   = os.getenv("R2_PUBLIC_BASE_URL",   "").rstrip("/")
R2_KEY_PREFIX        = os.getenv("R2_KEY_PREFIX",        "images").strip("/")

engine        = create_async_engine(DATABASE_URL, echo=False)
session_maker = async_sessionmaker(engine, expire_on_commit=False)

VALID_EXTENSIONS = {"jpg", "jpeg", "png"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_config():
    missing = [
        var for var, val in [
            ("R2_ENDPOINT_URL",      R2_ENDPOINT_URL),
            ("R2_ACCESS_KEY_ID",     R2_ACCESS_KEY_ID),
            ("R2_SECRET_ACCESS_KEY", R2_SECRET_ACCESS_KEY),
            ("R2_PUBLIC_BASE_URL",   R2_PUBLIC_BASE_URL),
        ] if not val
    ]
    if missing:
        print(f"ERROR: Missing required env vars: {', '.join(missing)}")
        sys.exit(1)


def build_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def r2_key(filename: str) -> str:
    prefix = f"{R2_KEY_PREFIX}/" if R2_KEY_PREFIX else ""
    return f"{prefix}{filename}"


def r2_public_url(filename: str) -> str:
    prefix = f"{R2_KEY_PREFIX}/" if R2_KEY_PREFIX else ""
    encoded = quote(filename, safe="")
    return f"{R2_PUBLIC_BASE_URL}/{prefix}{encoded}"


def r2_object_exists(client, key: str) -> bool:
    try:
        client.head_object(Bucket=R2_PUBLIC_BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def upload_to_r2(client, local_path: Path, key: str) -> None:
    ext = local_path.suffix.lower()
    content_type = "image/png" if ext == ".png" else "image/jpeg"
    with open(local_path, "rb") as f:
        client.put_object(
            Bucket=R2_PUBLIC_BUCKET,
            Key=key,
            Body=f,
            ContentType=content_type,
        )


def delete_from_r2(client, key: str) -> None:
    client.delete_object(Bucket=R2_PUBLIC_BUCKET, Key=key)


def swap_extension(filename: str, to_ext: str) -> str:
    """Replace the file extension. Normalises 'jpg'/'jpeg' → 'jpg'."""
    stem = Path(filename).stem
    return f"{stem}.{to_ext}"


def rename_chroma_id(collection, old_id: str, new_id: str) -> bool:
    """
    ChromaDB has no rename — fetch the entry, delete it, re-add with new id.
    Returns True if the old id existed and was renamed, False if not found.
    """
    result = collection.get(ids=[old_id], include=["embeddings", "documents", "metadatas"])

    if not result["ids"]:
        return False

    embedding = result["embeddings"][0] if result["embeddings"] else None
    document  = result["documents"][0]  if result["documents"]  else None
    metadata  = result["metadatas"][0]  if result["metadatas"]  else {}

    collection.delete(ids=[old_id])
    collection.add(
        ids=[new_id],
        embeddings=[embedding] if embedding is not None else None,
        documents=[document]   if document  is not None else None,
        metadatas=[metadata or {}],
    )
    return True


# ── Main logic ────────────────────────────────────────────────────────────────

async def run(
    to_ext: str,
    replace: bool,
    do_r2: bool,
    do_db: bool,
    do_chroma: bool,
    do_metadata: bool,
):
    from app.db.models import Image as DBImage

    if not METADATA_PATH.exists():
        print(f"ERROR: metadata file not found: {METADATA_PATH}")
        sys.exit(1)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    # Identify items that need updating (current extension ≠ target)
    to_process = [
        item for item in items
        if Path(item["filename"]).suffix.lstrip(".").lower().replace("jpeg", "jpg") != to_ext
    ]

    total = len(items)
    skip  = total - len(to_process)

    steps = []
    if do_r2:       steps.append("R2 upload" + (" + delete old" if replace else ""))
    if do_db:       steps.append("database")
    if do_chroma:   steps.append("ChromaDB")
    if do_metadata: steps.append("metadata.json")

    print(f"Loaded   : {total} entries from {METADATA_PATH}")
    print(f"Target   : .{to_ext}")
    print(f"Steps    : {', '.join(steps) if steps else 'NONE — nothing will be changed'}")
    print(f"Already .{to_ext}: {skip} items skipped")
    print(f"To update: {len(to_process)} items")
    print()

    if not to_process or not steps:
        print("Nothing to do.")
        return

    # Only initialise R2 client if needed
    client = build_r2_client() if do_r2 else None

    # Only open ChromaDB if needed
    chroma_col = None
    if do_chroma:
        from app.chromadb import CLIENT as chroma_client, COLLECTION_NAME
        chroma_col = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    uploaded       = 0
    db_updated     = 0
    r2_deleted     = 0
    chroma_renamed = 0
    missing        = 0
    errors         = 0

    async with session_maker() as db:
        for i, item in enumerate(to_process, 1):
            old_filename = item["filename"]
            new_filename = swap_extension(old_filename, to_ext)
            new_local    = IMAGES_FOLDER / new_filename
            old_key      = r2_key(old_filename)
            new_key      = r2_key(new_filename)
            new_url      = r2_public_url(new_filename)
            label        = f"[{i:>4}/{len(to_process)}] {old_filename} → {new_filename}"

            # ── R2: upload new file ───────────────────────────────────────
            if do_r2:
                if not new_local.exists():
                    print(f"  MISSING  {label}")
                    missing += 1
                    continue
                try:
                    upload_to_r2(client, new_local, new_key)
                    uploaded += 1
                except Exception as e:
                    print(f"  ERROR    {label} — R2 upload failed: {e}")
                    errors += 1
                    continue

            # ── Metadata: update in-memory ────────────────────────────────
            if do_metadata:
                item["filename"] = new_filename
                item["url"]      = new_url

            # ── DB: update filename + url ─────────────────────────────────
            if do_db:
                try:
                    result = await db.execute(
                        select(DBImage).where(DBImage.filename == old_filename)
                    )
                    db_image = result.scalar_one_or_none()
                    if db_image:
                        db_image.filename = new_filename
                        db_image.url      = new_url
                        await db.flush()
                        db_updated += 1
                    else:
                        print(f"  WARN     {label} — not found in DB")
                except Exception as e:
                    print(f"  ERROR    {label} — DB update failed: {e}")
                    errors += 1

            # ── ChromaDB: rename ID ───────────────────────────────────────
            if do_chroma:
                try:
                    found = rename_chroma_id(chroma_col, old_filename, new_filename)
                    if found:
                        chroma_renamed += 1
                    else:
                        print(f"  WARN     {label} — not found in ChromaDB")
                except Exception as e:
                    print(f"  ERROR    {label} — ChromaDB rename failed: {e}")

            # ── R2: delete old object if --replace ────────────────────────
            if do_r2 and replace:
                try:
                    if r2_object_exists(client, old_key):
                        delete_from_r2(client, old_key)
                        r2_deleted += 1
                    else:
                        print(f"  SKIP DEL {label} — old R2 object not found")
                except Exception as e:
                    print(f"  ERROR    {label} — R2 delete failed: {e}")

            print(f"  OK       {label}")

        if do_db:
            await db.commit()

    # ── Write updated metadata back to disk ──────────────────────────────────
    if do_metadata:
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    print()
    print("─" * 60)
    if do_r2:       print(f"Uploaded to R2 : {uploaded}")
    if do_db:       print(f"DB updated     : {db_updated}")
    if do_chroma:   print(f"ChromaDB IDs   : {chroma_renamed} renamed")
    if do_r2 and replace:
                    print(f"R2 deleted     : {r2_deleted}")
    if do_metadata: print(f"Metadata saved : {METADATA_PATH}")
    print(f"Missing locally: {missing}")
    print(f"Errors         : {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Sync a format change (png↔jpg) across metadata.json, database, ChromaDB, and R2.\n"
            "By default all four steps run. Use --only-* flags to run just specific steps."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python sync_image_format.py --to jpg                     # all steps\n"
            "  uv run python sync_image_format.py --to jpg --replace           # all steps + delete old R2\n"
            "  uv run python sync_image_format.py --to jpg --only-metadata     # metadata.json only\n"
            "  uv run python sync_image_format.py --to jpg --only-db           # database only\n"
            "  uv run python sync_image_format.py --to jpg --only-r2           # R2 upload only\n"
            "  uv run python sync_image_format.py --to jpg --only-r2 --replace # R2 upload + delete old\n"
            "  uv run python sync_image_format.py --to jpg --only-chroma       # ChromaDB only\n"
        ),
    )
    parser.add_argument(
        "--to",
        required=True,
        choices=["jpg", "png"],
        metavar="EXT",
        help="Target image extension: jpg or png",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="(R2 step) Delete old-format objects from R2 after uploading new ones.",
    )
    parser.add_argument("--only-r2",       action="store_true", help="Run R2 upload step only")
    parser.add_argument("--only-db",       action="store_true", help="Run database update step only")
    parser.add_argument("--only-chroma",   action="store_true", help="Run ChromaDB rename step only")
    parser.add_argument("--only-metadata", action="store_true", help="Run metadata.json update step only")

    args = parser.parse_args()

    # If any --only-* flag is set, run only those; otherwise run everything
    any_only = args.only_r2 or args.only_db or args.only_chroma or args.only_metadata
    do_r2       = args.only_r2       if any_only else True
    do_db       = args.only_db       if any_only else True
    do_chroma   = args.only_chroma   if any_only else True
    do_metadata = args.only_metadata if any_only else True

    if do_r2:
        validate_config()

    asyncio.run(run(
        to_ext=args.to,
        replace=args.replace,
        do_r2=do_r2,
        do_db=do_db,
        do_chroma=do_chroma,
        do_metadata=do_metadata,
    ))
