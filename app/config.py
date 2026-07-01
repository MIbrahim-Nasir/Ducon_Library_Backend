"""
app/config.py
─────────────
Central environment/config helpers shared across the app so security-sensitive
behaviour (fail-closed in production, CORS origins, signed URLs) is consistent
and defined in one place instead of duplicated per module.
"""
from __future__ import annotations

import os

# ── Environment ────────────────────────────────────────────────────────────────
ENV = os.getenv("ENV", os.getenv("ENVIRONMENT", "development")).strip().lower()
IS_PRODUCTION = ENV in ("production", "prod")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


# ── CORS ─────────────────────────────────────────────────────────────────────
# Comma-separated list of allowed origins. Falls back to common local dev ports
# when unset so local development keeps working without configuration.
# Common Vite / CRA dev ports (Ducon Library Vite uses 5174 — strictPort).
_DEV_PORTS = (3000, 4173, 5173, 5174, 5175)
_DEFAULT_DEV_ORIGINS = [
    origin
    for port in _DEV_PORTS
    for origin in (f"http://localhost:{port}", f"http://127.0.0.1:{port}")
]


def get_cors_origins() -> list[str]:
    configured = _split_csv(os.getenv("CORS_ALLOW_ORIGINS", ""))
    if configured:
        return configured
    # No explicit config: allow local dev origins only. In production this should
    # be set explicitly; an empty list here means "no cross-origin access".
    return [] if IS_PRODUCTION else _DEFAULT_DEV_ORIGINS
