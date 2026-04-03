"""
upload_images_to_r2.py
───────────────────────
Uploads images to a Cloudflare R2 public bucket and writes updated metadata.

Default mode
    Checks every image in metadata against the R2 bucket (HEAD request).
    Skips files already present (controlled by SKIP_EXISTING env var).
    Writes a NEW metadata file to OUTPUT_METADATA_PATH — original is unchanged.

    uv run python upload_images_to_r2.py

--fast mode
    Skips the R2 bucket check entirely.
    Only processes items that have no url or whose url is not a Cloudflare R2 url
    (i.e. doesn't start with R2_PUBLIC_BASE_URL).
    Updates the SOURCE metadata file in-place.

    uv run python upload_images_to_r2.py --fast

Options (via env vars):
    METADATA_PATH        Source metadata file  (default: data/images/metadata.json)
    IMAGES_FOLDER        Folder containing image files (default: data/images)
    R2_ENDPOINT_URL      https://<account-id>.r2.cloudflarestorage.com
    R2_ACCESS_KEY_ID     R2 API token access key
    R2_SECRET_ACCESS_KEY R2 API token secret key
    R2_PUBLIC_BUCKET     Bucket name (e.g. ducon-public)
    R2_PUBLIC_BASE_URL   Public URL for the bucket (e.g. https://pub-xxxx.r2.dev)
    R2_KEY_PREFIX        Optional path prefix inside the bucket (default: images)
    OUTPUT_METADATA_PATH Where to write the new metadata in default mode
                         (default: data/images/metadata.r2.json)
    SKIP_EXISTING        Skip files already in R2 in default mode (default: true)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
METADATA_PATH        = Path(os.getenv("METADATA_PATH",       "data/images/metadata.json"))
IMAGES_FOLDER        = Path(os.getenv("IMAGES_FOLDER",       "data/images"))
OUTPUT_METADATA_PATH = Path(os.getenv("OUTPUT_METADATA_PATH", "data/images/metadata.r2.json"))

R2_ENDPOINT_URL      = os.getenv("R2_ENDPOINT_URL",      "")
R2_ACCESS_KEY_ID     = os.getenv("R2_ACCESS_KEY_ID",     "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_PUBLIC_BUCKET     = os.getenv("R2_PUBLIC_BUCKET",     "ducon-public")
R2_PUBLIC_BASE_URL   = os.getenv("R2_PUBLIC_BASE_URL",   "").rstrip("/")
R2_KEY_PREFIX        = os.getenv("R2_KEY_PREFIX",        "images").strip("/")
SKIP_EXISTING        = os.getenv("SKIP_EXISTING", "true").lower() not in ("false", "0", "no")


def validate_config():
    missing = []
    for var, val in [
        ("R2_ENDPOINT_URL",      R2_ENDPOINT_URL),
        ("R2_ACCESS_KEY_ID",     R2_ACCESS_KEY_ID),
        ("R2_SECRET_ACCESS_KEY", R2_SECRET_ACCESS_KEY),
        ("R2_PUBLIC_BASE_URL",   R2_PUBLIC_BASE_URL),
    ]:
        if not val:
            missing.append(var)
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


def key_exists(client, bucket: str, key: str) -> bool:
    """Return True if the object already exists in R2."""
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def r2_url(filename: str) -> str:
    """Construct the public R2 URL for a given filename."""
    prefix = f"{R2_KEY_PREFIX}/" if R2_KEY_PREFIX else ""
    encoded = quote(filename, safe="")
    return f"{R2_PUBLIC_BASE_URL}/{prefix}{encoded}"


def upload_file(client, local_path: Path, bucket: str, key: str) -> None:
    """Upload a single file to R2 with correct content type."""
    content_type = "image/png" if local_path.suffix.lower() == ".png" else "image/jpeg"
    with open(local_path, "rb") as f:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=f,
            ContentType=content_type,
        )


def needs_upload(item: dict) -> bool:
    """
    Returns True if the item has no url or its url is not an R2 Cloudflare url.
    Used in --fast mode to skip items already pointing at R2.
    """
    url = item.get("url", "")
    return not url or not url.startswith(R2_PUBLIC_BASE_URL)


def run_default(items: list[dict], client) -> list[dict]:
    """
    Default mode: check R2 for each file, upload if missing (or SKIP_EXISTING=false).
    Returns a new list of items with updated urls. Writes to OUTPUT_METADATA_PATH.
    """
    uploaded  = 0
    skipped   = 0
    not_found = 0
    output_items = []

    for i, item in enumerate(items, 1):
        filename   = item["filename"]
        local_path = IMAGES_FOLDER / filename
        prefix     = f"{R2_KEY_PREFIX}/" if R2_KEY_PREFIX else ""
        r2_key     = f"{prefix}{filename}"
        label      = f"[{i:>4}/{len(items)}] {filename}"

        if not local_path.exists():
            print(f"  MISSING  {label}")
            not_found += 1
            output_items.append({**item})
            continue

        if SKIP_EXISTING and key_exists(client, R2_PUBLIC_BUCKET, r2_key):
            print(f"  SKIP     {label}")
            skipped += 1
        else:
            try:
                upload_file(client, local_path, R2_PUBLIC_BUCKET, r2_key)
                print(f"  UPLOADED {label}")
                uploaded += 1
            except Exception as e:
                print(f"  ERROR    {label} — {e}")
                output_items.append({**item})
                continue

        output_items.append({**item, "url": r2_url(filename)})

    OUTPUT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(output_items, f, ensure_ascii=False, indent=2)

    print()
    print("─" * 60)
    print(f"Uploaded : {uploaded}")
    print(f"Skipped  : {skipped}")
    print(f"Missing  : {not_found}")
    print(f"Written  : {OUTPUT_METADATA_PATH}")


def run_fast(items: list[dict], client) -> None:
    """
    Fast mode: no R2 bucket checks. Only uploads items that have no url or a
    non-R2 url. Updates the source metadata file in-place.
    """
    to_upload = [item for item in items if needs_upload(item)]

    print(f"{len(to_upload)} items need uploading (no url or non-R2 url).")
    print()

    uploaded  = 0
    not_found = 0

    for i, item in enumerate(to_upload, 1):
        filename   = item["filename"]
        local_path = IMAGES_FOLDER / filename
        prefix     = f"{R2_KEY_PREFIX}/" if R2_KEY_PREFIX else ""
        r2_key     = f"{prefix}{filename}"
        label      = f"[{i:>4}/{len(to_upload)}] {filename}"

        if not local_path.exists():
            print(f"  MISSING  {label}")
            not_found += 1
            continue

        try:
            upload_file(client, local_path, R2_PUBLIC_BUCKET, r2_key)
            item["url"] = r2_url(filename)
            print(f"  UPLOADED {label}")
            uploaded += 1
        except Exception as e:
            print(f"  ERROR    {label} — {e}")

    # Write back to the source metadata file
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print()
    print("─" * 60)
    print(f"Uploaded : {uploaded}")
    print(f"Missing  : {not_found}")
    print(f"Written  : {METADATA_PATH}  (in-place)")


def main():
    parser = argparse.ArgumentParser(description="Upload Ducon library images to Cloudflare R2.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Skip R2 bucket checks. Only upload items with no url or a non-R2 url. "
            "Updates the source metadata file in-place."
        ),
    )
    args = parser.parse_args()

    validate_config()

    if not METADATA_PATH.exists():
        print(f"ERROR: metadata file not found: {METADATA_PATH}")
        sys.exit(1)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    print(f"Loaded   : {len(items)} entries from {METADATA_PATH}")
    print(f"Bucket   : {R2_PUBLIC_BUCKET}")
    print(f"Prefix   : {R2_KEY_PREFIX or '(none)'}")
    print(f"Base URL : {R2_PUBLIC_BASE_URL}")
    print(f"Mode     : {'--fast (in-place, no R2 checks)' if args.fast else 'default (full check → new file)'}")
    print()

    client = build_r2_client()

    if args.fast:
        run_fast(items, client)
    else:
        run_default(items, client)


if __name__ == "__main__":
    main()
