"""Apply a permissive GET CORS policy to the Cloudflare R2 private bucket.

Why this exists
---------------
The Studio "continue to chat" flow downloads a generated image as a blob via
`fetch(presignedR2Url)` from the browser. R2 buckets have NO CORS policy by
default, so the browser blocks the request with:

    Access to fetch at 'https://<account>.r2.cloudflarestorage.com/ducon-private/...'
    from origin 'http://localhost:5174' has been blocked by CORS policy:
    No 'Access-Control-Allow-Origin' header is present on the requested resource.

Plain <img> tags work without CORS, but any `fetch()`-as-blob (downloads, chat
handoff, canvas reads) needs the bucket to return `Access-Control-Allow-Origin`.

This script writes a CORS policy onto the bucket using the same R2 credentials
the backend already uses (R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
R2_PRIVATE_BUCKET). It is idempotent — safe to re-run any time.

Usage
-----
    python -m scripts.set_r2_cors

    # Optional: override origins without editing .env
    R2_CORS_ORIGINS="http://localhost:5174,https://app.yourdomain.com" python -m scripts.set_r2_cors

After running, R2 serves `Access-Control-Allow-Origin` on GET responses and the
browser unblocks the fetch.
"""
import os
import sys

import boto3
from botocore.exceptions import ClientError


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
    access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    bucket = os.getenv("R2_PRIVATE_BUCKET", "ducon-private").strip()

    if not (endpoint and access_key and secret_key):
        print(
            "ERROR: R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY "
            "must be set (load your backend .env first).",
            file=sys.stderr,
        )
        return 1

    # Default to localhost dev origin + a production placeholder. Override via
    # R2_CORS_ORIGINS in .env (comma-separated).
    default_origins = "http://localhost:5174,http://127.0.0.1:5174"
    origins = _split_csv(os.getenv("R2_CORS_ORIGINS", default_origins))
    if not origins:
        print("ERROR: R2_CORS_ORIGINS resolved to an empty list.", file=sys.stderr)
        return 1

    cors_config = {
        "CORSRules": [
            {
                "AllowedOrigins": origins,
                "AllowedMethods": ["GET"],
                # R2 ignores most headers, but listing `*` is the documented way
                # to permit arbitrary request headers on the presigned GET.
                "AllowedHeaders": ["*"],
                "ExposeHeaders": [
                    "ETag",
                    "Content-Type",
                    "Content-Length",
                ],
                "MaxAgeSeconds": 3600,
            }
        ]
    }

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    print(f"Applying CORS policy to bucket '{bucket}' at {endpoint}")
    print(f"  Allowed origins: {origins}")
    try:
        client.put_bucket_cors(Bucket=bucket, CORSConfiguration=cors_config)
    except ClientError as exc:
        print(f"ERROR: failed to set CORS: {exc}", file=sys.stderr)
        return 2

    # Read it back to confirm.
    try:
        resp = client.get_bucket_cors(Bucket=bucket)
        rules = resp.get("CORSRules", [])
        print(f"OK — bucket now has {len(rules)} CORS rule(s):")
        for rule in rules:
            print(f"  origins={rule.get('AllowedOrigins')} "
                  f"methods={rule.get('AllowedMethods')}")
    except ClientError as exc:
        print(f"Set succeeded but read-back failed: {exc}", file=sys.stderr)

    print("\nNext: hard-refresh the browser (Ctrl+Shift+R) and retry continue-to-chat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
