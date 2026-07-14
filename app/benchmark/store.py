"""Local JSON persistence for the internal AI-generation benchmark dashboard.

No database, no Cloudflare — just files under ``benchmark_data/`` at the
backend repo root. All file I/O runs through ``asyncio.to_thread`` so the
event loop stays free for SSE streaming.

Layout
------
``benchmark_data/test_cases.json``  — list of saved test cases
``benchmark_data/combos.json``      — list of saved combos
``benchmark_data/runs/{run_group_id}.json`` — one file per run group
``benchmark_data/designer_jobs/{job_id}.json`` — persisted dev designer jobs
``benchmark_data/uploads/{uuid}.{ext}``     — uploaded image files
``benchmark_data/outputs/{run_group_id}/{process_id}/{n}.png`` — run outputs

The directory is in ``.gitignore`` so generated artefacts are never committed.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


# ── Paths ──────────────────────────────────────────────────────────────────────

# Repo root = parent of the `app/` package this module lives in.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _REPO_ROOT / "benchmark_data"
RUNS_DIR = DATA_DIR / "runs"
DESIGNER_JOBS_DIR = DATA_DIR / "designer_jobs"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
TEST_CASES_FILE = DATA_DIR / "test_cases.json"
COMBOS_FILE = DATA_DIR / "combos.json"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    DESIGNER_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Low-level JSON helpers (run in threads) ────────────────────────────────────

def _atomic_write(path: Path, payload: Any) -> None:
    _ensure_dirs()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_list(path: Path) -> list[dict]:
    data = _read_json(path)
    if isinstance(data, list):
        return data
    return []


def _write_list(path: Path, items: list[dict]) -> None:
    _atomic_write(path, items)


def _upsert_by_id(items: list[dict], item: dict) -> list[dict]:
    item_id = item.get("id")
    for i, existing in enumerate(items):
        if existing.get("id") == item_id:
            items[i] = item
            return items
    items.append(item)
    return items


# ── Test cases ─────────────────────────────────────────────────────────────────

def _normalize_test_case(tc: dict) -> dict:
    if not tc.get("id"):
        tc["id"] = str(uuid4())
    tc.setdefault("name", "")
    tc.setdefault("hint", "")
    tc.setdefault("use_ducon_data", False)
    inputs = tc.get("inputs") or []
    norm_inputs = []
    for inp in inputs:
        if not isinstance(inp, dict):
            continue
        ni = {
            "label": inp.get("label", ""),
            "role": inp.get("role", "user"),
            "source": inp.get("source", "upload"),
            "name": inp.get("name", ""),
        }
        if inp.get("catalog_id") is not None:
            ni["catalog_id"] = inp["catalog_id"]
        if inp.get("upload_id") is not None:
            ni["upload_id"] = inp["upload_id"]
        if inp.get("metadata") is not None:
            ni["metadata"] = inp["metadata"]
        norm_inputs.append(ni)
    tc["inputs"] = norm_inputs
    return tc


async def save_test_case(tc: dict) -> dict:
    tc = _normalize_test_case(dict(tc))
    items = await list_test_cases()
    items = _upsert_by_id(items, tc)
    await asyncio.to_thread(_write_list, TEST_CASES_FILE, items)
    return tc


async def list_test_cases() -> list[dict]:
    return await asyncio.to_thread(_read_list, TEST_CASES_FILE)


async def delete_test_case(test_case_id: str) -> bool:
    items = await list_test_cases()
    new_items = [t for t in items if t.get("id") != test_case_id]
    if len(new_items) == len(items):
        return False
    await asyncio.to_thread(_write_list, TEST_CASES_FILE, new_items)
    return True


async def get_test_case(test_case_id: str) -> Optional[dict]:
    items = await list_test_cases()
    return next((t for t in items if t.get("id") == test_case_id), None)


# ── Combos ─────────────────────────────────────────────────────────────────────

_COMBO_FIELDS = (
    "name", "flow", "image_model", "image_thinking",
    "prompt_model", "prompt_thinking", "max_eval_rounds",
    "max_prompt_verify_rounds", "system_prompt_override", "aspect_ratio",
    "image_model_pair", "prompt_model_pair",
    "designer_tool_access", "designer_system_prompt_mode", "designer_filesystem_root",
    "designer_max_generation_rounds", "designer_max_turns", "designer_wall_clock_budget_s",
)


def _normalize_combo(c: dict) -> dict:
    if not c.get("id"):
        c["id"] = str(uuid4())
    _NONE_FIELDS = ("system_prompt_override", "aspect_ratio")
    _INT_DEFAULTS = {
        "max_eval_rounds": 3,
        "max_prompt_verify_rounds": 2,
        "designer_max_generation_rounds": 5,
        "designer_max_turns": 24,
        "designer_wall_clock_budget_s": 15 * 60,
    }
    for field in _COMBO_FIELDS:
        if field in _NONE_FIELDS:
            c.setdefault(field, None)
        elif field in _INT_DEFAULTS:
            c.setdefault(field, _INT_DEFAULTS[field])
        elif field in ("image_model_pair", "prompt_model_pair"):
            c.setdefault(field, None)
        elif field == "designer_tool_access":
            c.setdefault(field, None)
        elif field in ("designer_system_prompt_mode", "designer_filesystem_root"):
            c.setdefault(field, None)
        else:
            c.setdefault(field, "")
    if isinstance(c.get("image_model_pair"), dict):
        c["image_model"] = str(c["image_model_pair"].get("model_id") or c.get("image_model") or "")
    elif c.get("image_model"):
        c["image_model_pair"] = {
            "router": "gemini_native",
            "model_id": c.get("image_model"),
            "id": f"gemini_native:{c.get('image_model')}",
        }
    if isinstance(c.get("prompt_model_pair"), dict):
        c["prompt_model"] = str(c["prompt_model_pair"].get("model_id") or c.get("prompt_model") or "")
    elif c.get("prompt_model"):
        c["prompt_model_pair"] = {
            "router": "gemini_native",
            "model_id": c.get("prompt_model"),
            "id": f"gemini_native:{c.get('prompt_model')}",
        }
    return c


async def save_combo(c: dict) -> dict:
    c = _normalize_combo(dict(c))
    items = await list_combos()
    items = _upsert_by_id(items, c)
    await asyncio.to_thread(_write_list, COMBOS_FILE, items)
    return c


async def list_combos() -> list[dict]:
    return await asyncio.to_thread(_read_list, COMBOS_FILE)


async def delete_combo(combo_id: str) -> bool:
    items = await list_combos()
    new_items = [c for c in items if c.get("id") != combo_id]
    if len(new_items) == len(items):
        return False
    await asyncio.to_thread(_write_list, COMBOS_FILE, new_items)
    return True


async def get_combo(combo_id: str) -> Optional[dict]:
    items = await list_combos()
    return next((c for c in items if c.get("id") == combo_id), None)


# ── Uploads ────────────────────────────────────────────────────────────────────

_EXT_WHITELIST = {"png", "jpg", "jpeg", "webp", "gif", "bmp", "heic", "heif"}


def _clean_ext(ext: str) -> str:
    ext = (ext or "").lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    if ext not in _EXT_WHITELIST:
        ext = "png"
    return ext


def _write_upload(file_bytes: bytes, upload_id: str, ext: str) -> str:
    _ensure_dirs()
    filename = f"{upload_id}.{ext}"
    path = UPLOADS_DIR / filename
    with open(path, "wb") as f:
        f.write(file_bytes)
    return filename


async def save_upload(file_bytes: bytes, ext: str, name: str = "") -> dict:
    ext = _clean_ext(ext)
    upload_id = str(uuid4())
    filename = await asyncio.to_thread(_write_upload, file_bytes, upload_id, ext)
    return {
        "id": upload_id,
        "url": f"/dev/uploads/{filename}",
        "name": name or filename,
    }


def upload_path(filename: str) -> Path:
    return UPLOADS_DIR / filename


def output_path(run_group_id: str, process_id: str, idx: int) -> Path:
    return OUTPUTS_DIR / run_group_id / process_id / f"{idx}.png"


def _write_output_png(run_group_id: str, process_id: str, idx: int, png_bytes: bytes) -> str:
    _ensure_dirs()
    path = output_path(run_group_id, process_id, idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(png_bytes)
    return f"/dev/outputs/{run_group_id}/{process_id}/{idx}.png"


async def save_output_image(run_group_id: str, process_id: str, idx: int, png_bytes: bytes) -> str:
    return await asyncio.to_thread(
        _write_output_png, run_group_id, process_id, idx, png_bytes
    )


# ── Run groups ─────────────────────────────────────────────────────────────────

def _run_group_path(run_group_id: str) -> Path:
    return RUNS_DIR / f"{run_group_id}.json"


def _normalize_run_group(rg: dict) -> dict:
    if not rg.get("id"):
        rg["id"] = str(uuid4())
    rg.setdefault("created_at", time.time())
    rg.setdefault("status", "running")
    rg.setdefault("test_case_ids", [])
    rg.setdefault("combo_ids", [])
    rg.setdefault("processes", [])
    return rg


async def save_run_group(rg: dict) -> dict:
    rg = _normalize_run_group(dict(rg))
    await asyncio.to_thread(_atomic_write, _run_group_path(rg["id"]), rg)
    return rg


async def get_run_group(run_group_id: str) -> Optional[dict]:
    return await asyncio.to_thread(_read_json, _run_group_path(run_group_id))


async def list_run_groups() -> list[dict]:
    files = await asyncio.to_thread(lambda: list(RUNS_DIR.glob("*.json")))
    out: list[dict] = []
    for f in files:
        try:
            data = await asyncio.to_thread(_read_json, f)
        except Exception:
            continue
        if isinstance(data, dict):
            out.append(data)
    out.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return out


def _delete_run_group_sync(run_group_id: str) -> bool:
    path = _run_group_path(run_group_id)
    if not path.exists():
        return False
    path.unlink()
    outputs_dir = OUTPUTS_DIR / run_group_id
    if outputs_dir.exists():
        shutil.rmtree(outputs_dir, ignore_errors=True)
    return True


async def delete_run_group(run_group_id: str) -> bool:
    return await asyncio.to_thread(_delete_run_group_sync, run_group_id)


# ── Designer jobs ──────────────────────────────────────────────────────────────

def _designer_job_path(job_id: str) -> Path:
    return DESIGNER_JOBS_DIR / f"{job_id}.json"


def _normalize_designer_job(job: dict) -> dict:
    if not job.get("id"):
        job["id"] = str(uuid4())
    job.setdefault("kind", "designer")
    job.setdefault("created_at", time.time())
    job.setdefault("status", "queued")
    job.setdefault("input", {})
    job.setdefault("events", [])
    job.setdefault("final", None)
    job.setdefault("error", None)
    return job


async def save_designer_job(job: dict) -> dict:
    job = _normalize_designer_job(dict(job))
    await asyncio.to_thread(_atomic_write, _designer_job_path(job["id"]), job)
    return job


async def get_designer_job(job_id: str) -> Optional[dict]:
    return await asyncio.to_thread(_read_json, _designer_job_path(job_id))


async def list_designer_jobs(*, active_only: bool = False) -> list[dict]:
    files = await asyncio.to_thread(lambda: list(DESIGNER_JOBS_DIR.glob("*.json")))
    out: list[dict] = []
    for f in files:
        try:
            data = await asyncio.to_thread(_read_json, f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if active_only and data.get("status") not in {"queued", "running"}:
            continue
        out.append(data)
    out.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return out


def _delete_designer_job_sync(job_id: str) -> bool:
    path = _designer_job_path(job_id)
    if not path.exists():
        return False
    path.unlink()
    outputs_dir = OUTPUTS_DIR / job_id
    if outputs_dir.exists():
        shutil.rmtree(outputs_dir, ignore_errors=True)
    return True


async def delete_designer_job(job_id: str) -> bool:
    return await asyncio.to_thread(_delete_designer_job_sync, job_id)
