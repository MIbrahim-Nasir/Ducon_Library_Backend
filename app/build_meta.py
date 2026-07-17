"""Build / deploy metadata exposed to clients for cache-busting."""
from __future__ import annotations

import os
from pathlib import Path


def get_build_id() -> str:
    """Stable identifier for the running backend deploy.

    Resolution order (first hit wins):
    1. Contents of the file at ``BUILD_ID_FILE`` — rewritten on each deploy and
       read on every call, so a graceful SIGHUP reload surfaces the new id
       WITHOUT a full restart. (Env vars are frozen at process start and would
       go stale after a reload.)
    2. ``APP_BUILD_ID`` env var — deploy-time fallback (git SHA, CI build number).
    3. ``dev`` — local development fallback.
    """
    build_id_file = os.getenv("BUILD_ID_FILE", "").strip()
    if build_id_file:
        try:
            value = Path(build_id_file).read_text(encoding="utf-8").strip()
            if value:
                return value
        except OSError:
            pass
    env_id = os.getenv("APP_BUILD_ID", "").strip()
    if env_id:
        return env_id
    return "dev"
