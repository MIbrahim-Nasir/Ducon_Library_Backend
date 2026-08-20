"""Syntax + critical-import smoke tests for the FastAPI app package.

Catches IndentationError / SyntaxError in modules that default pytest paths
never import (e.g. ``app.benchmark.designer_agent`` when the dev dashboard
router is not exercised). Prefer parsing source from disk over full
``app.main`` import so CI stays fast and does not need a complete runtime env.

Hardening notes (2026-07-23 recurrence):
- Compile via ``compile(source_bytes, ...)`` / ``ast.parse`` — never rely on
  ``.pyc`` timestamps or ``compileall(force=False)`` skip logic.
- Assert known-fragile paths are discovered and compiled; fail loudly if
  the file disappears from the tree or is unreadable.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"

# Modules that uvicorn / chat / cleanup can load at startup or on first use,
# including rarely-touched paths that broke production imports before.
CRITICAL_IMPORT_MODULES = (
    "app.benchmark.designer_agent",
    "app.designer_agent",
    "app.designer",
    "app.designer.runner",
    "app.chat_agent",
    "app.cleanup_scheduler",
    "app.tool_generate_image",
    "app.routers.multi_image_gen",
    "app.routers.internal_designer",
)

# Files that have repeatedly been left with IndentationError by bad edits.
# Must always be present under app/ and must always be compiled from disk.
MUST_COMPILE_REL_PATHS = (
    Path("app") / "benchmark" / "designer_agent.py",
    Path("app") / "designer_agent.py",
    Path("app") / "designer" / "runner.py",
    Path("app") / "designer" / "tools.py",
    Path("app") / "designer" / "loop.py",
    Path("app") / "gemini.py",
)


def _iter_app_py_files() -> list[Path]:
    return sorted(p for p in APP_DIR.rglob("*.py") if p.is_file())


def _compile_source_from_disk(path: Path) -> None:
    """Parse ``path`` from current disk bytes — no ``.pyc``, no stale cache.

    Raises SyntaxError / IndentationError (subclass of SyntaxError) on failure.
    """
    source = path.read_bytes()
    try:
        source_text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SyntaxError(f"cannot decode {path} as utf-8: {exc}") from exc
    # ast.parse + compile from disk bytes — never reads/writes .pyc
    ast.parse(source_text, filename=str(path))
    compile(source_text, str(path), "exec", dont_inherit=True)


def test_must_compile_paths_exist_on_disk():
    """Known-fragile modules must exist — empty/missing path must not silently pass."""
    missing = [str(rel) for rel in MUST_COMPILE_REL_PATHS if not (ROOT / rel).is_file()]
    assert not missing, "Required smoke paths missing on disk:\n" + "\n".join(missing)


def test_all_app_python_files_compile():
    """Every file under app/ must parse — no IndentationError / SyntaxError."""
    files = _iter_app_py_files()
    assert files, f"No Python files found under {APP_DIR}"

    rel_set = {p.relative_to(ROOT) for p in files}
    for required in MUST_COMPILE_REL_PATHS:
        assert required in rel_set, (
            f"{required.as_posix()} not discovered by app/ rglob — "
            f"smoke test would miss IndentationError there"
        )

    failures: list[str] = []
    for path in files:
        try:
            _compile_source_from_disk(path)
        except SyntaxError as exc:
            # IndentationError is a SyntaxError subclass
            failures.append(f"{path.relative_to(ROOT)}: {exc.__class__.__name__}: {exc}")

    assert not failures, "Syntax errors in app/:\n" + "\n".join(failures)


@pytest.mark.parametrize(
    "rel_path",
    MUST_COMPILE_REL_PATHS,
    ids=[p.as_posix() for p in MUST_COMPILE_REL_PATHS],
)
def test_known_fragile_file_compiles_from_disk(rel_path: Path):
    """Dedicated assert: these exact paths must compile from current disk bytes."""
    path = ROOT / rel_path
    assert path.is_file(), f"missing {rel_path.as_posix()}"
    _compile_source_from_disk(path)


@pytest.mark.parametrize("module_name", CRITICAL_IMPORT_MODULES)
def test_critical_modules_import(module_name: str):
    """Import critical modules so package-level failures surface in pytest."""
    mod = importlib.import_module(module_name)
    assert mod is not None
