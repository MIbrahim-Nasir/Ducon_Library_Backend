"""FastAPI dev router for the internal AI-generation benchmark dashboard.

Owns the HTTP layer + local persistence + SSE live streaming for the dev
benchmark dashboard. All routes are mounted under ``/dev`` and are dev-only:
in production the entire router fail-closes (404), and in dev it requires
either ``DEV_DASHBOARD_ENABLED`` to be truthy or a matching
``DEV_BENCHMARK_SECRET`` header (when a secret is configured).

Backend A owns ``app/benchmark/types.py`` and ``app/benchmark/runner.py``.
We import them defensively so this module loads even if Backend A's code is
still in flux; the runner is only invoked at request time.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from sqlalchemy import select

from app.config import IS_PRODUCTION
from app.db.database import async_session_maker
from app.db.models import Image as DBImage
from app.image_utils import load_ducon_image
from app.sse import SSE_HEADERS

from app.benchmark import designer_agent as dev_designer_agent
from app.benchmark import provider_registry, store

logger = logging.getLogger(__name__)


# ── Guarded runner import ──────────────────────────────────────────────────────
# Backend A is implementing these in parallel. Import defensively so this module
# loads cleanly even if the runner/types are incomplete or absent. The runner is
# only called at request time; if it's unavailable, /dev/runs returns 503.

try:  # noqa: BLE001 — intentional broad guard
    from app.benchmark.types import (
        BenchmarkConfig,
        BenchmarkInput,
        BenchmarkResult,
        BenchmarkStep,
    )
    from app.benchmark.runner import run_benchmark_case

    _RUNNER_AVAILABLE = True
    _RUNNER_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001
    BenchmarkConfig = None  # type: ignore[assignment]
    BenchmarkInput = None  # type: ignore[assignment]
    BenchmarkResult = None  # type: ignore[assignment]
    BenchmarkStep = None  # type: ignore[assignment]
    run_benchmark_case = None  # type: ignore[assignment]
    _RUNNER_AVAILABLE = False
    _RUNNER_IMPORT_ERROR = repr(exc)
    logger.warning(
        "[dev_benchmark] app.benchmark.runner not available yet: %s. "
        "/dev/runs will return 503 until Backend A finishes it.",
        _RUNNER_IMPORT_ERROR,
    )


# ── Access control ─────────────────────────────────────────────────────────────

def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


async def require_dev_access(request: Request) -> None:
    """Dev-only guard for every /dev route.

    - In production: always 404 (fail-closed; the mount guard in main.py also
      avoids registering the router, but this is the belt-and-suspenders check).
    - In dev: allow if DEV_DASHBOARD_ENABLED is truthy, OR if a
      DEV_BENCHMARK_SECRET header matches the configured env secret, OR if no
      secret is configured at all (open dev).
    """
    if IS_PRODUCTION:
        raise HTTPException(status_code=404)

    if _truthy(os.getenv("DEV_DASHBOARD_ENABLED", "")):
        return

    configured_secret = os.getenv("DEV_BENCHMARK_SECRET", "")
    if configured_secret:
        provided = request.headers.get("DEV_BENCHMARK_SECRET", "")
        if provided and provided == configured_secret:
            return
        raise HTTPException(status_code=403, detail="DEV_BENCHMARK_SECRET required")

    # No explicit enablement and no secret configured → open in dev only.
    return


router = APIRouter(prefix="/dev", tags=["dev-benchmark"], dependencies=[Depends(require_dev_access)])


# ── Catalog helpers ────────────────────────────────────────────────────────────

def _level_to_type(level: Any) -> Optional[str]:
    try:
        level = int(level)
    except (TypeError, ValueError):
        return None
    if level == 5:
        return "design"
    if level == 4:
        return "area"
    if level in (1, 2, 3):
        return "product"
    return None


def _load_catalog_metadata() -> list[dict]:
    """Reuse the existing catalog metadata loader from the images router."""
    try:
        from app.routers.images import _load_metadata
        data = _load_metadata()
        return data if isinstance(data, list) else []
    except Exception as exc:  # noqa: BLE001
        logger.warning("[dev_benchmark] catalog metadata unavailable: %s", exc)
        return []


def _catalog_entry(e: dict) -> dict:
    filename = e.get("filename")
    full_url = e.get("url") or (f"/public/images/{filename}" if filename else None)
    return {
        "id": e.get("id"),
        "name": e.get("name") or e.get("title") or filename,
        "type": _level_to_type(e.get("level")),
        "area": e.get("area"),
        "products": e.get("products") or e.get("related") or [],
        "thumb_url": full_url,
        "full_url": full_url,
        "level": e.get("level"),
        "filename": filename,
    }


@router.get("/catalog/images")
async def catalog_images() -> dict:
    raw = [_catalog_entry(e) for e in _load_catalog_metadata() if isinstance(e, dict)]
    # metadata.json can repeat the same id for different filenames — dedupe for UI.
    seen: set[tuple] = set()
    images: list[dict] = []
    for entry in raw:
        key = (entry.get("id"), entry.get("filename"))
        if key in seen:
            continue
        seen.add(key)
        images.append(entry)
    return {"images": images}


@router.get("/catalog/areas")
async def catalog_areas() -> dict:
    areas = [
        _catalog_entry(e)
        for e in _load_catalog_metadata()
        if isinstance(e, dict) and _level_to_type(e.get("level")) == "area"
    ]
    return {"areas": areas}


@router.get("/catalog/products")
async def catalog_products() -> dict:
    products = [
        _catalog_entry(e)
        for e in _load_catalog_metadata()
        if isinstance(e, dict) and _level_to_type(e.get("level")) == "product"
    ]
    return {"products": products}


# ── Models / routers ───────────────────────────────────────────────────────────

@router.get("/models")
async def models() -> dict:
    pairs = await provider_registry.list_model_pairs()
    serialized = [provider_registry.serialize_pair(pair) for pair in pairs]
    # Backwards-compatible flat list for older dashboard builds.
    flat = [
        {
            "id": pair.model_id,
            "router": pair.router,
            "pair_id": pair.id,
            "model_id": pair.model_id,
            "roles": pair.roles,
            "thinking_modes": pair.thinking_modes,
        }
        for pair in pairs
    ]
    all_modes = sorted({mode for pair in pairs for mode in pair.thinking_modes})
    return {
        "models": flat,
        "model_pairs": serialized,
        "routers": provider_registry.ROUTERS,
        "thinking_modes": all_modes,
    }


@router.post("/model-pairs")
async def create_model_pair(body: dict) -> dict:
    try:
        pair = await provider_registry.save_model_pair(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return provider_registry.serialize_pair(pair)


@router.delete("/model-pairs/{pair_id:path}")
async def delete_model_pair(pair_id: str) -> dict:
    deleted = await provider_registry.delete_model_pair(pair_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Custom model pair not found.")
    return {"deleted": True, "id": pair_id}


# ── Uploads + file serving ─────────────────────────────────────────────────────

@router.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Empty upload.")
    raw_name = file.filename or "upload.png"
    ext = raw_name.rsplit(".", 1)[-1] if "." in raw_name else "png"
    return await store.save_upload(data, ext, name=raw_name)


@router.get("/uploads/{path:path}")
async def serve_upload(path: str) -> FileResponse:
    p = store.upload_path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Upload not found.")
    return FileResponse(str(p))


@router.get("/outputs/{run_group_id}/{process_id}/{n}.png")
async def serve_output(run_group_id: str, process_id: str, n: str) -> FileResponse:
    try:
        idx = int(n)
    except ValueError:
        raise HTTPException(status_code=404, detail="Invalid output index.")
    p = store.output_path(run_group_id, process_id, idx)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Output image not found.")
    return FileResponse(str(p), media_type="image/png")


# ── Reveal in OS file manager ──────────────────────────────────────────────────

def _resolve_dev_url(url: str) -> Optional[Path]:
    """Resolve a ``/dev/outputs/...`` or ``/dev/uploads/...`` URL to a disk path.

    Returns the resolved Path if it stays inside ``benchmark_data/``; returns
    None for unrecognized prefixes or for paths that escape via ``..`` or
    absolute-path segments.
    """
    raw = (url or "").strip()
    if raw.startswith("/"):
        raw = raw[1:]
    if raw.startswith("dev/outputs/"):
        candidate = store.OUTPUTS_DIR / raw[len("dev/outputs/"):]
    elif raw.startswith("dev/uploads/"):
        candidate = store.UPLOADS_DIR / raw[len("dev/uploads/"):]
    else:
        return None
    data_dir = store.DATA_DIR.resolve()
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError):
        return None
    try:
        if not resolved.is_relative_to(data_dir):
            return None
    except AttributeError:  # pragma: no cover — Python <3.9
        try:
            resolved.relative_to(data_dir)
        except ValueError:
            return None
    return resolved


async def _open_in_file_manager(path: Path) -> None:
    """Fire-and-forget launch of the OS file manager at ``path``.

    Picks the per-OS reveal command and runs it via ``create_subprocess_exec``.
    We do not await completion (file managers detach and stay open), but we wait
    briefly to surface immediate launch failures like "command not found".
    """
    absolute = str(path.resolve())
    system = platform.system()
    if system == "Windows":
        cmd = ["explorer", f"/select,{absolute}"]
    elif system == "Darwin":
        cmd = ["open", "-R", absolute]
    else:
        cmd = ["xdg-open", str(path.resolve().parent)]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        # Still launching — expected for fire-and-forget file managers.
        pass


@router.post("/reveal")
async def reveal_path(body: dict) -> dict:
    url = str(body.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url required")
    disk_path = _resolve_dev_url(url)
    if disk_path is None:
        raise HTTPException(status_code=403, detail="path outside benchmark_data")
    if not disk_path.exists() or not disk_path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    try:
        await _open_in_file_manager(disk_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "path": str(disk_path)}


# ── Dev designer agent ────────────────────────────────────────────────────────

def _parse_json_form(value: Optional[str], fallback: Any) -> Any:
    if value is None or value == "":
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON form field: {exc}") from exc


@router.get("/designer/config")
async def designer_config() -> dict:
    return dev_designer_agent.get_designer_config()


@router.get("/session-config-schema")
async def session_config_schema_endpoint() -> dict:
    """Session/context field metadata: limits, descriptions, router applicability."""
    from app.benchmark.session_config import session_config_schema

    return session_config_schema()


@router.post("/designer/jobs")
async def create_designer_job(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    model_pair_id: str = Form(""),
    image_pair_id: str = Form(""),
    thinking: str = Form(""),
    image_thinking: str = Form(""),
    system_prompt: str = Form(""),
    system_prompt_mode: str = Form("append"),
    system_prompt_sections: str = Form(""),
    tool_access: str = Form(""),
    filesystem_root: str = Form(""),
    aspect_ratio: str = Form("auto"),
    max_generation_rounds: str = Form(""),
    max_turns: str = Form(""),
    wall_clock_budget_s: str = Form(""),
    max_tokens: str = Form(""),
    eval_max_tokens: str = Form(""),
    max_messages: str = Form(""),
    max_tool_result_chars: str = Form(""),
    retain_recent_image_turns: str = Form(""),
    context_token_budget: str = Form(""),
    context_trigger_ratio: str = Form(""),
    context_policy: str = Form(""),
    openrouter_context_compression: str = Form(""),
    claude_compaction: str = Form(""),
    claude_compaction_trigger_tokens: str = Form(""),
    claude_compaction_instructions: str = Form(""),
    summarizer_mode: str = Form(""),
    summarizer_instructions: str = Form(""),
    max_eval_rounds: str = Form(""),
    max_prompt_verify_rounds: str = Form(""),
) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Designer image upload is empty.")
    tools_raw = _parse_json_form(tool_access, list(dev_designer_agent.DEFAULT_DESIGNER_TOOLS))
    tools = {str(item) for item in tools_raw} if isinstance(tools_raw, list) else set()
    if not tools:
        tools = set(dev_designer_agent.DEFAULT_DESIGNER_TOOLS)
    sections_raw = _parse_json_form(system_prompt_sections, {})
    section_overrides = sections_raw if isinstance(sections_raw, dict) else {}
    mode = (system_prompt_mode or "append").strip().lower()
    if mode not in {"append", "replace", "compose"}:
        raise HTTPException(status_code=422, detail="system_prompt_mode must be append, replace, or compose.")

    default_rounds = str(dev_designer_agent.DEFAULT_MAX_GENERATION_ROUNDS)
    default_turns = str(dev_designer_agent.DEFAULT_MAX_TURNS)
    try:
        max_rounds = int((max_generation_rounds or default_rounds).strip() or default_rounds)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="max_generation_rounds must be a positive integer."
        ) from exc
    if max_rounds < 1:
        raise HTTPException(status_code=422, detail="max_generation_rounds must be >= 1.")
    try:
        max_turns_val = int((max_turns or default_turns).strip() or default_turns)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="max_turns must be an integer (0 = unlimited).") from exc
    if max_turns_val < 0:
        raise HTTPException(status_code=422, detail="max_turns must be >= 0 (0 = unlimited).")
    default_wall = str(dev_designer_agent.WALL_CLOCK_BUDGET_S)
    try:
        wall_clock_val = int((wall_clock_budget_s or default_wall).strip() or default_wall)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="wall_clock_budget_s must be an integer (0 = unlimited)."
        ) from exc
    if wall_clock_val < 0:
        raise HTTPException(status_code=422, detail="wall_clock_budget_s must be >= 0 (0 = unlimited).")

    model_pair = await provider_registry.get_model_pair(model_pair_id or "gemini_native:gemini-3-flash-preview")
    image_pair = await provider_registry.get_model_pair(
        image_pair_id or "gemini_native:gemini-3-pro-image-preview"
    )
    if thinking and thinking not in model_pair.thinking_modes:
        raise HTTPException(status_code=422, detail="Invalid thinking mode for selected model pair.")
    if image_thinking and image_thinking not in image_pair.thinking_modes:
        raise HTTPException(status_code=422, detail="Invalid image thinking mode for selected image pair.")

    ext = (file.filename or "upload.png").rsplit(".", 1)[-1] if file.filename else "png"
    upload = await store.save_upload(data, ext, file.filename or "upload")

    session_form = {
        k: v
        for k, v in {
            "max_tokens": max_tokens,
            "eval_max_tokens": eval_max_tokens,
            "max_messages": max_messages,
            "max_tool_result_chars": max_tool_result_chars,
            "retain_recent_image_turns": retain_recent_image_turns,
            "context_token_budget": context_token_budget,
            "context_trigger_ratio": context_trigger_ratio,
            "context_policy": context_policy,
            "openrouter_context_compression": openrouter_context_compression,
            "claude_compaction": claude_compaction,
            "claude_compaction_trigger_tokens": claude_compaction_trigger_tokens,
            "claude_compaction_instructions": claude_compaction_instructions or None,
            "summarizer_mode": summarizer_mode,
            "summarizer_instructions": summarizer_instructions or None,
            "max_eval_rounds": max_eval_rounds,
            "max_prompt_verify_rounds": max_prompt_verify_rounds,
        }.items()
        if v is not None and str(v).strip() != ""
    }

    input_meta = {
        "prompt": prompt,
        "upload_url": upload["url"],
        "upload_name": upload["name"],
        "aspect_ratio": aspect_ratio or dev_designer_agent.DEFAULT_ASPECT_RATIO,
        "model_pair": provider_registry.serialize_pair(model_pair),
        "image_pair": provider_registry.serialize_pair(image_pair),
        "thinking": thinking or (model_pair.thinking_modes[0] if model_pair.thinking_modes else None),
        "image_thinking": image_thinking or (image_pair.thinking_modes[0] if image_pair.thinking_modes else None),
        "system_prompt": system_prompt or None,
        "system_prompt_mode": mode,
        "tool_access": sorted(tools),
        "filesystem_root": filesystem_root or None,
        "max_generation_rounds": max_rounds,
        "max_turns": max_turns_val,
        "wall_clock_budget_s": wall_clock_val,
        **session_form,
    }
    job = dev_designer_agent.create_job(input_meta=input_meta)
    await dev_designer_agent.persist_job(job)
    job.task = asyncio.create_task(
        dev_designer_agent.run_dev_designer_job(
            job=job,
            user_image_bytes=data,
            user_prompt=prompt,
            model_pair=model_pair,
            image_pair=image_pair,
            thinking=thinking or (model_pair.thinking_modes[0] if model_pair.thinking_modes else None),
            image_thinking=image_thinking or (image_pair.thinking_modes[0] if image_pair.thinking_modes else None),
            system_prompt=system_prompt or None,
            system_prompt_mode=mode,
            system_prompt_sections=section_overrides or None,
            tool_access=tools,
            filesystem_root=filesystem_root or None,
            aspect_ratio=aspect_ratio or None,
            max_generation_rounds=max_rounds,
            max_turns=max_turns_val,
            wall_clock_budget_s=wall_clock_val,
        )
    )
    return {
        "job_id": job.id,
        "status": job.status,
        "model_pair": provider_registry.serialize_pair(model_pair),
        "image_pair": provider_registry.serialize_pair(image_pair),
    }


@router.get("/designer/jobs")
async def list_designer_jobs(active_only: bool = False) -> dict:
    jobs = await store.list_designer_jobs(active_only=active_only)
    summaries = []
    for j in jobs:
        inp = j.get("input") or {}
        summaries.append(
            {
                "id": j.get("id"),
                "kind": "designer",
                "status": j.get("status"),
                "created_at": j.get("created_at"),
                "prompt": inp.get("prompt") or "",
                "upload_url": inp.get("upload_url"),
                "upload_name": inp.get("upload_name"),
                "error": j.get("error"),
            }
        )
    return {"designer_jobs": summaries}


@router.get("/designer/jobs/{job_id}")
async def get_designer_job(job_id: str) -> dict:
    job = dev_designer_agent.get_job(job_id)
    if job is None:
        snapshot = await store.get_designer_job(job_id)
        if not isinstance(snapshot, dict):
            raise HTTPException(status_code=404, detail="Designer job not found.")
        return snapshot
    return dev_designer_agent.job_to_snapshot(job)


@router.delete("/designer/jobs/{job_id}")
async def delete_designer_job(job_id: str) -> dict:
    deleted = await store.delete_designer_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Designer job not found.")
    dev_designer_agent.JOBS.pop(job_id, None)
    return {"deleted": True, "id": job_id}


@router.post("/designer/jobs/{job_id}/cancel")
async def cancel_designer_job(job_id: str) -> dict:
    ok = await dev_designer_agent.cancel_job(job_id)
    if not ok:
        job = dev_designer_agent.get_job(job_id)
        if job is None:
            snapshot = await store.get_designer_job(job_id)
            if not isinstance(snapshot, dict):
                raise HTTPException(status_code=404, detail="Designer job not found.")
            status = str(snapshot.get("status") or "")
            if status in {"completed", "failed", "cancelled"}:
                raise HTTPException(status_code=409, detail=f"Job already {status}.")
            raise HTTPException(status_code=404, detail="Designer job not found.")
        if job.status in {"completed", "failed", "cancelled"}:
            raise HTTPException(status_code=409, detail=f"Job already {job.status}.")
        raise HTTPException(status_code=409, detail="Job cannot be cancelled.")
    return {"cancelled": True, "id": job_id}


@router.get("/designer/jobs/{job_id}/stream")
async def stream_designer_job(job_id: str):
    job = dev_designer_agent.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Designer job not found.")
    return StreamingResponse(
        dev_designer_agent.stream_job(job),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# ── Test cases CRUD ────────────────────────────────────────────────────────────

@router.post("/test-cases")
async def create_test_case(body: dict) -> dict:
    return await store.save_test_case(body)


@router.get("/test-cases")
async def list_test_cases() -> dict:
    return {"test_cases": await store.list_test_cases()}


@router.delete("/test-cases/{test_case_id}")
async def delete_test_case(test_case_id: str) -> dict:
    deleted = await store.delete_test_case(test_case_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Test case not found.")
    return {"deleted": True, "id": test_case_id}


# ── Combos CRUD ────────────────────────────────────────────────────────────────

async def _normalize_combo_pairs(body: dict) -> dict:
    c = dict(body)
    image_ref = c.get("image_model_pair") or c.get("image_pair") or c.get("image_model")
    prompt_ref = c.get("prompt_model_pair") or c.get("prompt_pair") or c.get("prompt_model")
    image_pair = await provider_registry.get_model_pair(image_ref)
    prompt_pair = await provider_registry.get_model_pair(prompt_ref)
    c["image_model_pair"] = provider_registry.serialize_pair(image_pair)
    c["prompt_model_pair"] = provider_registry.serialize_pair(prompt_pair)
    c["image_model"] = image_pair.model_id
    c["prompt_model"] = prompt_pair.model_id

    image_thinking = c.get("image_thinking")
    if image_thinking not in image_pair.thinking_modes:
        c["image_thinking"] = image_pair.thinking_modes[0]
    prompt_thinking = c.get("prompt_thinking")
    if prompt_thinking not in prompt_pair.thinking_modes:
        c["prompt_thinking"] = prompt_pair.thinking_modes[0]
    return c

@router.post("/combos")
async def create_combo(body: dict) -> dict:
    return await store.save_combo(await _normalize_combo_pairs(body))


@router.get("/combos")
async def list_combos() -> dict:
    return {"combos": await store.list_combos()}


@router.delete("/combos/{combo_id}")
async def delete_combo(combo_id: str) -> dict:
    deleted = await store.delete_combo(combo_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Combo not found.")
    return {"deleted": True, "id": combo_id}


# ── Run orchestration: live state + SSE ────────────────────────────────────────


class _RunGroupLive:
    """In-memory live state for one run group, shared by HTTP + SSE handlers."""

    __slots__ = (
        "id", "created_at", "test_case_ids", "combo_ids",
        "processes", "subscribers", "lock", "tasks", "status",
    )

    def __init__(
        self,
        run_group_id: str,
        created_at: float,
        test_case_ids: list[str],
        combo_ids: list[str],
        processes: list[dict],
    ) -> None:
        self.id = run_group_id
        self.created_at = created_at
        self.test_case_ids = test_case_ids
        self.combo_ids = combo_ids
        # preserve insertion order so the snapshot list is stable
        self.processes: dict[str, dict] = {p["id"]: p for p in processes}
        self.subscribers: list[asyncio.Queue] = []
        self.lock = asyncio.Lock()
        self.tasks: list[asyncio.Task] = []
        self.status = "running"


_RUNS: dict[str, _RunGroupLive] = {}
_PERSIST_LAST: dict[str, float] = {}
_PERSIST_DEBOUNCE_S = 0.5


def _build_snapshot(rg: _RunGroupLive) -> dict:
    return {
        "id": rg.id,
        "created_at": rg.created_at,
        "status": rg.status,
        "test_case_ids": list(rg.test_case_ids),
        "combo_ids": list(rg.combo_ids),
        "processes": [
            {
                "id": pid,
                "test_case_id": p["test_case_id"],
                "combo_id": p["combo_id"],
                "status": p["status"],
                "current_step": p.get("current_step"),
                "steps": p.get("steps", []),
                "result": p.get("result"),
            }
            for pid, p in rg.processes.items()
        ],
    }


async def _emit(rg: _RunGroupLive, event: dict) -> None:
    for q in list(rg.subscribers):
        try:
            q.put_nowait(event)
        except Exception:  # noqa: BLE001 — never let a slow client kill a run
            pass


async def _persist_run_group(run_group_id: str, debounce: bool = False) -> None:
    rg = _RUNS.get(run_group_id)
    if rg is None:
        return
    if debounce:
        last = _PERSIST_LAST.get(run_group_id, 0.0)
        if time.time() - last < _PERSIST_DEBOUNCE_S:
            return
    _PERSIST_LAST[run_group_id] = time.time()
    async with rg.lock:
        snapshot = _build_snapshot(rg)
    await store.save_run_group(snapshot)


def _serialize_step(step: Any) -> dict:
    if isinstance(step, dict):
        return step
    try:
        return dataclasses.asdict(step)
    except Exception:  # noqa: BLE001
        return {
            "index": getattr(step, "index", None),
            "kind": getattr(step, "kind", None),
            "model": getattr(step, "model", None),
            "thinking": getattr(step, "thinking", None),
            "status": getattr(step, "status", None),
            "started_at": getattr(step, "started_at", None),
            "ended_at": getattr(step, "ended_at", None),
            "duration_ms": getattr(step, "duration_ms", None),
            "prompt_used": getattr(step, "prompt_used", None),
            "tokens_in": getattr(step, "tokens_in", None),
            "tokens_out": getattr(step, "tokens_out", None),
            "image_count": getattr(step, "image_count", None),
            "cost_usd": getattr(step, "cost_usd", None),
            "error": getattr(step, "error", None),
        }


async def _serialize_result(
    result: Any, run_group_id: str, process_id: str, test_case: Optional[dict] = None
) -> dict:
    try:
        d = dataclasses.asdict(result)
    except Exception:  # noqa: BLE001
        d = {
            "process_id": getattr(result, "process_id", None),
            "config": getattr(result, "config", None),
            "inputs_summary": getattr(result, "inputs_summary", None),
            "steps": getattr(result, "steps", None),
            "output_images": getattr(result, "output_images", None),
            "final_prompt": getattr(result, "final_prompt", None),
            "retries": getattr(result, "retries", None),
            "total_duration_ms": getattr(result, "total_duration_ms", None),
            "total_cost_usd": getattr(result, "total_cost_usd", None),
            "cost_breakdown": getattr(result, "cost_breakdown", None),
            "status": getattr(result, "status", None),
            "error": getattr(result, "error", None),
        }

    # Serialize the config dataclass if asdict didn't already produce a dict.
    if not isinstance(d.get("config"), dict) and d.get("config") is not None:
        try:
            d["config"] = dataclasses.asdict(d["config"])
        except Exception:  # noqa: BLE001
            pass

    # Replace raw PNG bytes with saved-file URLs.
    raw_imgs = d.get("output_images") or []
    urls: list[dict] = []
    for i, png in enumerate(raw_imgs):
        if png:
            url = await store.save_output_image(run_group_id, process_id, i, png)
        else:
            url = f"/dev/outputs/{run_group_id}/{process_id}/{i}.png"
        urls.append({"index": i, "url": url})
    d["output_images"] = urls

    # Input image URLs for the results / process detail UI.
    input_images: list[dict] = []
    for inp in (test_case or {}).get("inputs") or []:
        if not isinstance(inp, dict):
            continue
        source = inp.get("source")
        meta = inp.get("metadata") if isinstance(inp.get("metadata"), dict) else {}
        url: Optional[str] = None
        if source == "upload":
            upload_id = inp.get("upload_id")
            if upload_id:
                name = inp.get("name") or meta.get("filename") or "upload.png"
                ext = name.rsplit(".", 1)[-1] if "." in name else "png"
                url = f"/dev/uploads/{upload_id}.{ext}"
        elif source == "catalog":
            url = meta.get("full_url") or meta.get("thumb_url") or meta.get("url")
            filename = meta.get("filename") or inp.get("name")
            if not url and filename:
                url = f"/public/images/{filename}"
        if url:
            input_images.append(
                {
                    "url": url,
                    "role": inp.get("role", "user"),
                    "metadata": meta or None,
                }
            )
    d["input_images"] = input_images

    retries = d.get("retries")
    if not retries:
        step_list = d.get("steps") or []
        retries = sum(
            1
            for s in step_list
            if isinstance(s, dict)
            and s.get("kind") == "prompt_retry"
            and s.get("status") == "completed"
        )
    d["retries"] = int(retries or 0)

    return d


# ── Input resolution ───────────────────────────────────────────────────────────

def _metadata_by_filename() -> dict[str, dict]:
    return {
        e.get("filename"): e
        for e in _load_catalog_metadata()
        if isinstance(e, dict) and e.get("filename")
    }


async def _load_upload_image(upload_id: str) -> Image.Image:
    def _load() -> Optional[Image.Image]:
        for p in store.UPLOADS_DIR.glob(f"{upload_id}.*"):
            try:
                return Image.open(p)
            except Exception:  # noqa: BLE001
                return None
        return None

    img = await asyncio.to_thread(_load)
    if img is None:
        raise FileNotFoundError(f"upload {upload_id} not found")
    return img


async def _load_catalog_image(catalog_id: int) -> tuple[Image.Image, Optional[str]]:
    async with async_session_maker() as db:
        result = await db.execute(select(DBImage).where(DBImage.id == catalog_id))
        row = result.scalar_one_or_none()
        if not row:
            raise FileNotFoundError(f"catalog image {catalog_id} not found")
        img = await load_ducon_image(row)
        return img, row.filename


async def _resolve_inputs(test_case: dict) -> list[Any]:
    use_ducon = bool(test_case.get("use_ducon_data"))
    meta_index = _metadata_by_filename() if use_ducon else {}
    out: list[Any] = []

    for inp in test_case.get("inputs") or []:
        if not isinstance(inp, dict):
            continue
        source = inp.get("source")
        meta: Optional[dict] = None
        try:
            if source == "upload":
                img = await _load_upload_image(inp.get("upload_id", ""))
            elif source == "catalog":
                img, filename = await _load_catalog_image(int(inp.get("catalog_id")))
                if use_ducon and filename:
                    meta = meta_index.get(filename)
            else:
                logger.warning("[dev_benchmark] unknown input source %r", source)
                continue
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422,
                detail=f"Failed to resolve input '{inp.get('label')}': {exc}",
            ) from exc

        out.append(
            BenchmarkInput(
                label=inp.get("label", ""),
                role=inp.get("role", "user"),
                pil_image=img,
                metadata=meta,
            )
        )
    return out


async def _handle_step(
    run_group_id: str, process_id: str, step: Any
) -> None:
    rg = _RUNS.get(run_group_id)
    if rg is None:
        return
    step_dict = _serialize_step(step)
    async with rg.lock:
        p = rg.processes.get(process_id)
        if p is None:
            return
        steps = p.setdefault("steps", [])
        idx = step_dict.get("index")
        kind = step_dict.get("kind")
        for i, s in enumerate(steps):
            if s.get("index") == idx and s.get("kind") == kind and idx is not None:
                steps[i] = step_dict
                break
        else:
            steps.append(step_dict)
        p["current_step"] = step_dict
        if step_dict.get("status") == "running":
            p["status"] = "running"

    await _persist_run_group(run_group_id, debounce=True)
    ev_type = "step_started" if step_dict.get("status") == "running" else "step_completed"
    await _emit(
        rg,
        {
            "type": ev_type,
            "run_group_id": run_group_id,
            "process_id": process_id,
            "step": step_dict,
        },
    )


async def _run_process(
    run_group_id: str,
    process_id: str,
    test_case: Optional[dict],
    combo: Optional[dict],
) -> None:
    rg = _RUNS.get(run_group_id)
    if rg is None:
        return

    async with rg.lock:
        rg.processes[process_id]["status"] = "running"

    await _emit(
        rg,
        {
            "type": "process_started",
            "run_group_id": run_group_id,
            "process_id": process_id,
            "test_case_id": test_case.get("id") if test_case else None,
            "combo_id": combo.get("id") if combo else None,
        },
    )

    try:
        if test_case is None or combo is None:
            raise RuntimeError("Test case or combo no longer exists.")
        if not _RUNNER_AVAILABLE:
            raise RuntimeError(f"Benchmark runner unavailable: {_RUNNER_IMPORT_ERROR}")

        inputs = await _resolve_inputs(test_case)
        image_pair = await provider_registry.get_model_pair(
            combo.get("image_model_pair") or combo.get("image_model")
        )
        prompt_pair = await provider_registry.get_model_pair(
            combo.get("prompt_model_pair") or combo.get("prompt_model")
        )
        config = BenchmarkConfig(
            flow=combo.get("flow") or "agent_loop",
            image_model=image_pair.model_id,
            image_thinking=combo.get("image_thinking") or "",
            prompt_model=prompt_pair.model_id,
            prompt_thinking=combo.get("prompt_thinking") or "",
            max_eval_rounds=int(combo.get("max_eval_rounds") or 3),
            max_prompt_verify_rounds=int(combo.get("max_prompt_verify_rounds") or 2),
            system_prompt_override=combo.get("system_prompt_override"),
            aspect_ratio=combo.get("aspect_ratio"),
            name=combo.get("name") or "",
            image_router=image_pair.router,
            prompt_router=prompt_pair.router,
            image_model_pair=provider_registry.serialize_pair(image_pair),
            prompt_model_pair=provider_registry.serialize_pair(prompt_pair),
            designer_tool_access=tuple(combo.get("designer_tool_access") or []) or None,
            designer_system_prompt_mode=combo.get("designer_system_prompt_mode"),
            designer_filesystem_root=combo.get("designer_filesystem_root"),
            designer_max_generation_rounds=int(
                combo.get("designer_max_generation_rounds")
                or dev_designer_agent.DEFAULT_MAX_GENERATION_ROUNDS
            ),
            designer_max_turns=(
                int(combo["designer_max_turns"])
                if combo.get("designer_max_turns") is not None
                else dev_designer_agent.DEFAULT_MAX_TURNS
            ),
            designer_wall_clock_budget_s=(
                int(combo["designer_wall_clock_budget_s"])
                if combo.get("designer_wall_clock_budget_s") is not None
                else dev_designer_agent.WALL_CLOCK_BUDGET_S
            ),
        )

        async def on_step(step: Any) -> None:
            await _handle_step(run_group_id, process_id, step)

        result = await run_benchmark_case(
            process_id=process_id,
            config=config,
            inputs=inputs,
            hint=test_case.get("hint") or None,
            use_ducon_data=bool(test_case.get("use_ducon_data")),
            on_step=on_step,
            run_group_id=run_group_id,
        )

        result_dict = await _serialize_result(
            result, run_group_id, process_id, test_case=test_case
        )
        status = getattr(result, "status", None) or (
            "completed" if not getattr(result, "error", None) else "failed"
        )

        async with rg.lock:
            rg.processes[process_id]["status"] = status
            rg.processes[process_id]["result"] = result_dict

        await _persist_run_group(run_group_id)

        if status == "completed":
            await _emit(
                rg,
                {
                    "type": "process_completed",
                    "run_group_id": run_group_id,
                    "process_id": process_id,
                    "result": result_dict,
                },
            )
        else:
            await _emit(
                rg,
                {
                    "type": "process_failed",
                    "run_group_id": run_group_id,
                    "process_id": process_id,
                    "error": getattr(result, "error", None),
                    "result": result_dict,
                },
            )

    except Exception as exc:  # noqa: BLE001
        logger.exception("[dev_benchmark] process %s failed", process_id)
        err = str(exc)
        async with rg.lock:
            rg.processes[process_id]["status"] = "failed"
            rg.processes[process_id]["result"] = {
                "status": "failed",
                "error": err,
                "output_images": [],
            }
        await _persist_run_group(run_group_id)
        await _emit(
            rg,
            {
                "type": "process_failed",
                "run_group_id": run_group_id,
                "process_id": process_id,
                "error": err,
                "result": None,
            },
        )


async def _supervise_run(run_group_id: str) -> None:
    rg = _RUNS.get(run_group_id)
    if rg is None:
        return
    if rg.tasks:
        await asyncio.gather(*rg.tasks, return_exceptions=True)
    async with rg.lock:
        all_done = all(
            p["status"] in ("completed", "failed") for p in rg.processes.values()
        )
        rg.status = "completed" if all_done else "running"
        final_status = rg.status
        snapshot = _build_snapshot(rg)
    # final flush (force, no debounce)
    _PERSIST_LAST.pop(run_group_id, None)
    await store.save_run_group(snapshot)
    await _emit(
        rg,
        {
            "type": "run_completed",
            "run_group_id": run_group_id,
            "status": final_status,
        },
    )
    # Keep _RUNS entry around briefly so late SSE subscribers / GET /runs can
    # see the final snapshot; it will be evicted on the next run or by process
    # restart. Memory cost is negligible for a dev tool.


# ── Run endpoints ──────────────────────────────────────────────────────────────

@router.post("/runs")
async def create_run(body: dict) -> dict:
    if not _RUNNER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "Benchmark runner not available yet (Backend A "
                f"app/benchmark/runner.py incomplete): {_RUNNER_IMPORT_ERROR}"
            ),
        )

    test_case_ids: list[str] = [str(x) for x in (body.get("test_case_ids") or [])]
    combo_ids: list[str] = [str(x) for x in (body.get("combo_ids") or [])]

    if not test_case_ids or not combo_ids:
        raise HTTPException(
            status_code=422,
            detail="test_case_ids and combo_ids must each contain at least one id.",
        )

    all_tcs = await store.list_test_cases()
    all_combos = await store.list_combos()
    tc_map = {t["id"]: t for t in all_tcs if t["id"] in test_case_ids}
    combo_map = {c["id"]: c for c in all_combos if c["id"] in combo_ids}

    missing_tcs = [i for i in test_case_ids if i not in tc_map]
    missing_cbs = [i for i in combo_ids if i not in combo_map]
    if missing_tcs or missing_cbs:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown test_cases={missing_tcs} or combos={missing_cbs}.",
        )

    run_group_id = str(uuid4())
    created_at = time.time()

    processes: list[dict] = []
    pairs: list[tuple[str, str, str]] = []
    for tc_id in test_case_ids:
        for cb_id in combo_ids:
            pid = str(uuid4())
            processes.append(
                {
                    "id": pid,
                    "test_case_id": tc_id,
                    "combo_id": cb_id,
                    "status": "pending",
                    "current_step": None,
                    "steps": [],
                    "result": None,
                }
            )
            pairs.append((pid, tc_id, cb_id))

    # Persist a stub run-group file immediately so /history and /results see it.
    await store.save_run_group(
        {
            "id": run_group_id,
            "created_at": created_at,
            "status": "running",
            "test_case_ids": test_case_ids,
            "combo_ids": combo_ids,
            "processes": [{**p} for p in processes],
        }
    )

    rg = _RunGroupLive(
        run_group_id=run_group_id,
        created_at=created_at,
        test_case_ids=test_case_ids,
        combo_ids=combo_ids,
        processes=processes,
    )
    _RUNS[run_group_id] = rg

    # Start all processes concurrently — do NOT block the request.
    for pid, tc_id, cb_id in pairs:
        task = asyncio.create_task(
            _run_process(run_group_id, pid, tc_map[tc_id], combo_map[cb_id])
        )
        rg.tasks.append(task)

    # Supervisor marks the run group complete once every process settles.
    asyncio.create_task(_supervise_run(run_group_id))

    return {
        "run_group_id": run_group_id,
        "processes": [
            {"id": p["id"], "test_case_id": p["test_case_id"], "combo_id": p["combo_id"]}
            for p in processes
        ],
    }


def _snapshot_from_live_or_disk(run_group_id: str) -> tuple[Optional[dict], bool]:
    """Return (snapshot, is_live). Used by GET /runs and GET /history/{id}."""
    rg = _RUNS.get(run_group_id)
    if rg is not None:
        snap = _build_snapshot(rg)
        return snap, True
    return None, False


@router.get("/runs/{run_group_id}")
async def get_run(run_group_id: str) -> dict:
    rg = _RUNS.get(run_group_id)
    if rg is not None:
        async with rg.lock:
            return _build_snapshot(rg)
    data = await store.get_run_group(run_group_id)
    if not data:
        raise HTTPException(status_code=404, detail="Run group not found.")
    return data


@router.get("/runs/{run_group_id}/stream")
async def stream_run(run_group_id: str):
    rg = _RUNS.get(run_group_id)

    if rg is None:
        # Run already finished (or never existed). Replay a terminal event.
        data = await store.get_run_group(run_group_id)
        if not data:
            raise HTTPException(status_code=404, detail="Run group not found.")

        async def replay() -> Any:
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "snapshot",
                        "run_group_id": run_group_id,
                        "snapshot": data,
                    }
                )
                + "\n\n"
            )
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "run_completed",
                        "run_group_id": run_group_id,
                        "status": data.get("status"),
                    }
                )
                + "\n\n"
            )

        return StreamingResponse(
            replay(), media_type="text/event-stream", headers=SSE_HEADERS
        )

    queue: asyncio.Queue = asyncio.Queue()
    rg.subscribers.append(queue)

    async def gen():
        try:
            async with rg.lock:
                initial = _build_snapshot(rg)
                already_done = rg.status in ("completed", "failed")
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "snapshot",
                        "run_group_id": run_group_id,
                        "snapshot": initial,
                    }
                )
                + "\n\n"
            )

            # Late subscriber: the run already finished before we joined, so the
            # run_completed event was emitted to queues that no longer include
            # ours. Emit the terminal event immediately and close.
            if already_done:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "run_completed",
                            "run_group_id": run_group_id,
                            "status": rg.status,
                        }
                    )
                    + "\n\n"
                )
                return

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                yield "data: " + json.dumps(event) + "\n\n"

                if event.get("type") == "run_completed":
                    break
        except asyncio.CancelledError:
            # Client disconnected — drop our subscription and bail out cleanly.
            raise
        finally:
            try:
                rg.subscribers.remove(queue)
            except ValueError:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.get("/runs/{run_group_id}/results")
async def get_run_results(run_group_id: str) -> dict:
    rg = _RUNS.get(run_group_id)
    if rg is not None and rg.status == "running":
        raise HTTPException(status_code=409, detail="Run group is still running.")

    data = await store.get_run_group(run_group_id)
    if not data:
        raise HTTPException(status_code=404, detail="Run group not found.")
    return data


@router.get("/history")
async def history() -> dict:
    groups = await store.list_run_groups()
    out = []
    for g in groups:
        out.append(
            {
                "id": g.get("id"),
                "kind": "benchmark",
                "created_at": g.get("created_at"),
                "test_case_count": len(g.get("test_case_ids") or []),
                "combo_count": len(g.get("combo_ids") or []),
                "status": g.get("status"),
            }
        )
    designer_jobs = await store.list_designer_jobs()
    designer_out = []
    for j in designer_jobs:
        inp = j.get("input") or {}
        designer_out.append(
            {
                "id": j.get("id"),
                "kind": "designer",
                "created_at": j.get("created_at"),
                "status": j.get("status"),
                "prompt": inp.get("prompt") or "",
                "upload_url": inp.get("upload_url"),
                "upload_name": inp.get("upload_name"),
                "error": j.get("error"),
            }
        )
    return {"run_groups": out, "designer_jobs": designer_out}


@router.get("/history/{run_group_id}")
async def history_detail(run_group_id: str) -> dict:
    return await get_run(run_group_id)


@router.delete("/history/{run_group_id}")
async def delete_history_run(run_group_id: str) -> dict:
    deleted = await store.delete_run_group(run_group_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run group not found.")
    _RUNS.pop(run_group_id, None)
    _PERSIST_LAST.pop(run_group_id, None)
    return {"deleted": True, "id": run_group_id}
