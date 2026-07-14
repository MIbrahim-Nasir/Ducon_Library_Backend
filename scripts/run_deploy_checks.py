from __future__ import annotations

import argparse
import compileall
import importlib
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTS = [
    "tests/test_tool_generate_image_regressions.py",
    "tests/test_multi_image_endpoint_deploy.py",
    "tests/test_chat_tools_deploy.py",
    "tests/test_designer_job_cross_worker.py",
    "tests/test_designer_job_cleanup.py",
    "tests/test_signup_multistep_deploy.py",
]
IMPORT_SANITY_MODULES = [
    "app.tool_generate_image",
    "app.routers.multi_image_gen",
]
CHECKLIST = ROOT / "tests" / "DEPLOYMENT_ENDPOINT_CHECKLIST.md"


def _print_step(title: str) -> None:
    print(f"\n== {title} ==")


def _compile_path(path: Path) -> bool:
    if not path.exists():
        print(f"Missing path: {path}")
        return False
    return compileall.compile_dir(str(path), quiet=1, force=False)


def run_syntax_checks() -> bool:
    _print_step("Syntax checks")
    ok = True
    for relative in ("app", "tests", "scripts"):
        path = ROOT / relative
        print(f"Compiling {relative}/")
        ok = _compile_path(path) and ok
    return ok


def run_import_checks() -> bool:
    _print_step("Import sanity checks")
    ok = True
    sys.path.insert(0, str(ROOT))
    for module_name in IMPORT_SANITY_MODULES:
        try:
            importlib.import_module(module_name)
            print(f"Imported {module_name}")
        except Exception as exc:
            ok = False
            print(f"FAILED importing {module_name}: {exc!r}")
    return ok


def run_pytest(pytest_args: list[str]) -> int:
    _print_step("Pytest deploy subset")
    cmd = [sys.executable, "-m", "pytest", *pytest_args]
    print("Running:", " ".join(cmd))
    try:
        completed = subprocess.run(cmd, cwd=ROOT, check=False)
    except FileNotFoundError as exc:
        print(f"Could not start pytest: {exc}")
        return 1
    return completed.returncode


def print_manual_checks() -> None:
    _print_step("Manual/env-dependent checklist")
    print(f"See {CHECKLIST.relative_to(ROOT)} for endpoint coverage still requiring staging services.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run practical pre-deploy checks for the FastAPI backend.",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Optional pytest args. Defaults to the deploy-focused test subset.",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Only run syntax/import checks.",
    )
    return parser.parse_args()


def main() -> int:
    os.chdir(ROOT)
    args = parse_args()

    syntax_ok = run_syntax_checks()
    import_ok = run_import_checks()

    pytest_rc = 0
    if not args.skip_pytest:
        pytest_args = args.pytest_args or DEFAULT_TESTS
        pytest_rc = run_pytest(pytest_args)

    print_manual_checks()

    if syntax_ok and import_ok and pytest_rc == 0:
        _print_step("Deploy checks passed")
        return 0

    _print_step("Deploy checks failed")
    if not syntax_ok:
        print("Syntax checks failed.")
    if not import_ok:
        print("Import sanity checks failed.")
    if pytest_rc != 0:
        print(f"Pytest exited with code {pytest_rc}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
