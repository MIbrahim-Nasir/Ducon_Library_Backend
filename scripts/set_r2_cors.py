"""Apply a GET CORS policy to the Cloudflare R2 private bucket.

Why this exists
---------------
The Studio UI downloads generation images as blobs via `fetch(presignedR2Url)`
(and via same-origin `/generations/{id}/image` which 302-redirects to R2).
R2 buckets have NO CORS policy by default, so the browser blocks the request:

    Access to fetch at 'https://<account>.r2.cloudflarestorage.com/ducon-private/...'
    from origin 'https://app.duconodl.com' has been blocked by CORS policy:
    No 'Access-Control-Allow-Origin' header is present on the requested resource.

Plain <img> tags work without CORS, but any `fetch()`-as-blob (thumbnails,
downloads, chat handoff, canvas reads) needs the bucket to return
`Access-Control-Allow-Origin`.

Credentialed fetches
--------------------
Never set AllowedOrigins to `*` if any client uses `credentials: 'include'` —
browsers reject `ACAO: *` on credentialed cross-origin responses. The frontend
blob helpers (`apiGetBlob`, `getGenerationBlobUrl`) use `credentials: 'omit'`
for R2 / redirect-followed image loads (auth is Bearer or signed query tokens).
Even so, prefer explicit SPA origins over `*` so only known fronts can read
private-bucket responses from JS.

This script writes a CORS policy onto the bucket using the same R2 credentials
the backend already uses (R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
R2_PRIVATE_BUCKET). It is idempotent — safe to re-run any time.

Usage
-----
    # From the backend venv, with .env loaded (or export R2_* first):
    set -a && source .env && set +a   # bash; or export vars manually on Windows
    python -m scripts.set_r2_cors

    # Production + local Vite (recommended):
    R2_CORS_ORIGINS="https://app.duconodl.com,http://localhost:5174,http://127.0.0.1:5174" \\
      python -m scripts.set_r2_cors

After running, hard-refresh the browser (Ctrl+Shift+R) and retry generation
thumbnails / ExperimentsStrip.
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

    # Explicit origins only — never default to "*". Override via R2_CORS_ORIGINS.
    default_origins = (
        "https://app.duconodl.com,"
        "http://localhost:5174,"
        "http://127.0.0.1:5174"
    )
    origins = _split_csv(os.getenv("R2_CORS_ORIGINS", default_origins))
    if not origins:
        print("ERROR: R2_CORS_ORIGINS resolved to an empty list.", file=sys.stderr)
        return 1
    if any(o == "*" for o in origins):
        print(
            "ERROR: AllowedOrigins must not include '*'. "
            "Wildcard ACAO breaks credentialed fetches and is overly broad. "
            "Pass explicit origins via R2_CORS_ORIGINS.",
            file=sys.stderr,
        )
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

    print("\nNext: hard-refresh the browser (Ctrl+Shift+R) and retry generation images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
