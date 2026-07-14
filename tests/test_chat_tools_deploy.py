"""Deploy regression tests for chat tool availability (designer job auth gate)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_APP_JSX = ROOT.parent / "Ducon_Library" / "src" / "App.jsx"


def _tool_names(*, user_id: int | None) -> list[str]:
    from app.chat_agent import get_chat_tools

    return [t["name"] for t in get_chat_tools(user_id=user_id)]


def test_get_chat_tools_guest_omits_start_designer_job():
    """Regression: guests must not receive start_designer_job in chat tools."""
    names = _tool_names(user_id=None)
    assert "start_designer_job" not in names
    assert "generate_multi_image" in names


def test_get_chat_tools_authenticated_includes_start_designer_job():
    """Regression: signed-in users must receive start_designer_job in chat tools."""
    names = _tool_names(user_id=42)
    assert "start_designer_job" in names
    assert "generate_multi_image" in names


def test_app_start_designer_job_useeffect_includes_user_dep():
    """Regression: bridge handler must depend on `user` to avoid stale signed-in rejection."""
    if not FRONTEND_APP_JSX.is_file():
        pytest.skip(f"Frontend repo not found at {FRONTEND_APP_JSX}")

    source = FRONTEND_APP_JSX.read_text(encoding="utf-8")
    match = re.search(
        r"bridge\._register\('start_designer_job'.*?"
        r"bridge\._unregister\('start_designer_job'\);\s*"
        r"\},\s*\[(.*?)\]\);",
        source,
        re.DOTALL,
    )
    assert match, "Could not locate start_designer_job useEffect dependency array"
    deps = [part.strip().strip("'\"") for part in match.group(1).split(",") if part.strip()]
    assert "user" in deps, f"start_designer_job useEffect deps must include user, got {deps!r}"
