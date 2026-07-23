"""Syntax + critical-import smoke tests for the FastAPI app package.

Catches IndentationError / SyntaxError in modules that default pytest paths
never import (e.g. ``app.benchmark.designer_agent`` when the dev dashboard
router is not exercised). Prefer ``compile()`` over full ``app.main`` import
so CI stays fast and does not need a complete runtime env.
"""

from __future__ import annotations

import importlib
import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"

# Modules that uvicorn / chat / cleanup can load at startup or on first use,
# including rarely-touched paths that broke production imports before.
CRITICAL_IMPORT_MODULES = (
    "app.benchmark.designer_agent",
    "app.designer_agent",
    "app.chat_agent",
    "app.cleanup_scheduler",
    "app.tool_generate_image",
    "app.routers.multi_image_gen",
)


def _iter_app_py_files() -> list[Path]:
    return sorted(p for p in APP_DIR.rglob("*.py") if p.is_file())


def test_all_app_python_files_compile():
    """Every file under app/ must parse — no IndentationError / SyntaxError."""
    files = _iter_app_py_files()
    assert files, f"No Python files found under {APP_DIR}"

    failures: list[str] = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc.msg}")

    assert not failures, "Syntax errors in app/:\n" + "\n".join(failures)


@pytest.mark.parametrize("module_name", CRITICAL_IMPORT_MODULES)
def test_critical_modules_import(module_name: str):
    """Import critical modules so package-level failures surface in pytest."""
    mod = importlib.import_module(module_name)
    assert mod is not None
