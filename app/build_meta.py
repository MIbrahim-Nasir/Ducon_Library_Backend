"""Build / deploy metadata exposed to clients for cache-busting."""
from __future__ import annotations

import os

_BUILD_ID = os.getenv("APP_BUILD_ID", "").strip()


def get_build_id() -> str:
    """Stable identifier for the running backend deploy.

    Set ``APP_BUILD_ID`` at deploy time (git SHA, CI build number, etc.).
    Falls back to ``dev`` so local development keeps working without config.
    """
    if _BUILD_ID:
        return _BUILD_ID
    return "dev"
