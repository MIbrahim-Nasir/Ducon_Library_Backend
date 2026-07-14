"""Dev benchmark designer agent.

This is a dashboard-scoped runner: it avoids production user/job tables, streams
progress through the /dev router, and writes generated outputs under
benchmark_data.  The production designer agent remains unchanged.
"""
from __future__ import annotations

import asyncio
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from uuid import uuid4

from PIL import Image

from app import chromadb, prompt_loader
from app.benchmark import provider_clients, provider_registry, store
from app.benchmark.filesystem_tool import FILESYSTEM_OPERATIONS, FilesystemScopeError, ScopedFilesystemTool
from app.benchmark.provider_registry import ModelRouterPair
from app.benchmark.session_config import parse_session_config, session_config_defaults
from app.benchmark.session_manager import (
    append_tool_message,
    maybe_compact_messages,
)
from app.catalog_keyword_search import keyword_search_catalog
from app.db.database import async_session_maker
from app.designer_agent import (
    _fallback_generation_prompt,
    _fallback_queries,
    _filename_candidates,
    _single_filename_candidates,
)
from app.image_utils import get_image_metadata, load_ducon_image, normalize_user_image
from app.ml import GeminiEmbeddingModel
from app.tool_generate_image import ImageDescriptor, generate_multi_image
from app.db.models import Image as DBImage
from sqlalchemy import select


DEFAULT_DESIGNER_TOOL_ACCESS = (
    "ai_search",
    "keyword_search",
    "get_image",
    "generate_multi_image",
    "filesystem",
)
DESIGNER_TOOL_CATALOG = DEFAULT_DESIGNER_TOOL_ACCESS + ("generate_multi_image_pipeline",)
# Backward-compatible alias used when tool_access is omitted.
DEFAULT_DESIGNER_TOOLS = DEFAULT_DESIGNER_TOOL_ACCESS
DEFAULT_PIPELINE_MAX_EVAL_ROUNDS = 3
DEFAULT_PIPELINE_PROMPT_VERIFY_ROUNDS = 2
SEARCH_LIMIT = 5
MAX_SOURCES = 6
MAX_CONTEXT_CHARS = 48_000
# Loop budgets (guards, not scripts — the agent decides when to stop within them).
DEFAULT_MAX_GENERATION_ROUNDS = 5   # cap on generate_multi_image calls per job
DEFAULT_MAX_TURNS = 24              # cap on model turns in the agent loop (0 = unlimited)
DEFAULT_ASPECT_RATIO = "auto"       # Gemini ImageConfig auto — match client photo
WALL_CLOCK_BUDGET_S = 15 * 60       # hard wall-clock stop (0 = unlimited)
MAX_IMAGES_PER_GET = 4


def _resolve_max_turns(max_turns: Optional[int]) -> Optional[int]:
    """Return model-turn cap, or None when unlimited (max_turns <= 0)."""
    if max_turns is None:
        return DEFAULT_MAX_TURNS
    if int(max_turns) <= 0:
        return None
    return max(1, int(max_turns))


def _resolve_wall_clock_budget_s(budget_s: Optional[int]) -> Optional[int]:
    """Return wall-clock cap in seconds, or None when unlimited (budget_s <= 0)."""
    if budget_s is None:
        return WALL_CLOCK_BUDGET_S
    if int(budget_s) <= 0:
        return None
    return max(1, int(budget_s))


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# Dev-only replacement for the production image-gen-agent system prompt section.
# The production ``app/prompts/image-gen-agent.md`` describes a legacy two-image
# ``combine_images`` pipeline. The dev designer agent is a tool-calling loop that
# uses ONLY ``generate_multi_image``, so we override that section inline here.
# The production .md file is left untouched.
DEV_IMAGE_GEN_SECTION = """\
You generate Ducon outdoor designs using generate_multi_image and/or
generate_multi_image_pipeline (whichever tools are enabled for this job).
There is NO two-image / combine path in this agent — ignore any legacy two-image
combine instructions you may have seen elsewhere.

generate_multi_image — fast path: one Gemini image call using the prompt you write.
Counts as one generation toward your budget.

generate_multi_image_pipeline — full workflow (when enabled): ImageGenAgent expands
your brief, pre-generation prompt verify, then generate → eval → retry until approved
or rounds exhausted. Same image_refs; slower but higher quality. Also counts as one
generation toward your budget.

Shared image_refs for both tools:
- "user_photo" — the client's space photo. ALWAYS include it first (camera lock +
  preservation anchor).
- "catalog:<id>" — a Ducon catalog reference found via search (design direction,
  then products). Load them with get_image first so you can see what you reference.
- "generation:<n>" — one of your previous generated candidates, for refinement.
- Write the generation prompt using the role-based, extract-from-images discipline:
  shortest labels, no colour/texture adjectives, name zones + preservation + camera lock.

Refinement: to revise a candidate, call generate_multi_image again with
["user_photo", "generation:<n>", ...catalog refs...] and a prompt that names the
specific defects to fix while preserving what already works. Use evaluate_design
to get an independent judgement of a candidate before deciding to refine or stop.

Aspect ratio: default is ``auto`` (Gemini matches the client space photo). Override
only when the brief explicitly calls for a different framing (e.g. ``16:9``).
Do not specify aspect ratio in prompt text — it is set via the tool/config.
"""

# Agentic-loop header: persistence + stop criteria guidance (Claude Code / Cursor
# style: the model owns the loop; budgets are guards, not a script).
DEV_CATALOG_CONTEXT_SECTION = """\
Ducon catalog context (reference only — not a tool list):
Ducon is a UAE premium outdoor living company. The catalog holds completed project photos
and product references: pools, terraces, pergolas, pavers, landscaping, outdoor kitchens,
villa entrances, driveways, pathways, majlis seating, and decorative hardscape.

When picking catalog references, prefer full outdoor scenes that could apply to the client's
actual space over isolated product macro shots. Honor the user's brief on space type and style.

Important: this dev designer agent does NOT expose shortlist_directions, submit_directions,
or inspect_designs. Those tools belong to the production Studio Directions wizard (Step 4).
Use only the tools declared in your API tool list (ai_search, keyword_search, get_image,
generate_multi_image, generate_multi_image_pipeline, evaluate_design, fs_*, finish).
"""

DEV_AGENT_LOOP_SECTION = """\
You are an autonomous Ducon design agent operating in a tool-calling loop.
You are given a client's space photo and an optional brief. Your goal: produce a
photorealistic Ducon redesign of that space that you are confident in.

How to work:
1. Study the photo. Think about zones, fixed architecture, camera, and the brief.
2. Research the Ducon catalog with ai_search / keyword_search. Run as many
   searches as you need — different zones, materials, products, styles. Inspect
   promising hits with get_image before using them as references.
3. Generate with generate_multi_image (fast) or generate_multi_image_pipeline
   (full prompt+eval loop), user_photo first, then your chosen refs.
4. Judge the result yourself and/or call evaluate_design. If it has fixable
   defects and budget remains, refine (include "generation:<n>" in image_refs).
5. When you have a candidate you would present to a client, call finish with a
   short summary and the generation index you are selecting.

Persistence: keep going until the job is done well — do not stop after one
generation if it has clear defects and you still have budget. You decide when
the result is good enough; finish is YOUR call, within the budgets shown in
each tool result.

Filesystem: you have scoped file tools (fs_list, fs_read_text, fs_write_text,
fs_mkdir, fs_delete) rooted at a scratch directory. Use them when useful — e.g.
keep notes.md with your plan and findings, or write a design-brief.md for the
client. Nothing is written for you automatically.

Stop conditions (enforced by the harness): max model turns, max generations,
and a wall-clock budget. Each generate_multi_image result tells you how much
budget remains. If a tool errors repeatedly, work around it or finish with the
best candidate so far.
"""

# Short, focused evaluator system instruction for the per-round self-eval call.
_DEV_DESIGNER_EVAL_SYSTEM = (
    "You are a strict but practical visual quality gate for Ducon autonomous design previews.\n"
    "Compare the AI-generated candidate against the original client space photo and the design intent.\n"
    'Return STRICT JSON only: {"approved": boolean, "reasons": [string], "defects": [string]}.\n'
    "Set approved=true only if the redesign faithfully applies the requested Ducon design to the\n"
    "correct zones while preserving fixed architecture, camera viewpoint, and scene boundaries.\n"
    "defects lists concrete issues to fix in the next refinement round (empty when approved)."
)


@dataclass
class DevDesignerJob:
    id: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    queue: asyncio.Queue[dict[str, Any] | None] = field(default_factory=asyncio.Queue)
    task: Optional[asyncio.Task] = None
    final: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    input: Optional[dict[str, Any]] = None
    persist: bool = True
    cancel_requested: bool = False
    on_event: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None


JOBS: dict[str, DevDesignerJob] = {}


def job_to_snapshot(job: DevDesignerJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "kind": "designer",
        "status": job.status,
        "created_at": job.created_at,
        "input": job.input or {},
        "events": job.events,
        "final": job.final,
        "error": job.error,
    }


async def persist_job(job: DevDesignerJob) -> None:
    await store.save_designer_job(job_to_snapshot(job))


def create_job(*, input_meta: Optional[dict[str, Any]] = None) -> DevDesignerJob:
    job = DevDesignerJob(id=str(uuid4()), input=input_meta or {})
    JOBS[job.id] = job
    return job


def resolve_job_filesystem_root(filesystem_root: Optional[str], job_id: str) -> Optional[str]:
    """Return a per-job scratch subdirectory so each run starts with a clean scope."""
    if not filesystem_root or not str(filesystem_root).strip():
        return None
    base = Path(filesystem_root).expanduser()
    job_dir = base / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return str(job_dir.resolve(strict=False))


async def cancel_job(job_id: str) -> bool:
    """Request cancellation of a queued or running designer job."""
    job = await get_job_async(job_id)
    if job is None:
        return False
    if job.status not in {"queued", "running"}:
        return False
    job.cancel_requested = True
    task = job.task
    if task is not None and not task.done():
        task.cancel()
    return True


def _job_from_snapshot(data: dict[str, Any]) -> DevDesignerJob:
    job = DevDesignerJob(
        id=str(data["id"]),
        status=str(data.get("status") or "queued"),
        created_at=float(data.get("created_at") or time.time()),
        events=list(data.get("events") or []),
        final=data.get("final"),
        error=data.get("error"),
        input=data.get("input") or {},
    )
    return job


async def get_job_async(job_id: str) -> Optional[DevDesignerJob]:
    job = JOBS.get(job_id)
    if job is not None:
        return job
    data = await store.get_designer_job(job_id)
    if not isinstance(data, dict):
        return None
    job = _job_from_snapshot(data)
    JOBS[job_id] = job
    return job


def get_job(job_id: str) -> Optional[DevDesignerJob]:
    job = JOBS.get(job_id)
    if job is not None:
        return job
    path = store.DESIGNER_JOBS_DIR / f"{job_id}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            snapshot = json.load(f)
    except Exception:
        return None
    if not isinstance(snapshot, dict):
        return None
    job = _job_from_snapshot(snapshot)
    JOBS[job_id] = job
    return job


async def emit(job: DevDesignerJob, event_type: str, **data: Any) -> None:
    payload = {
        "type": event_type,
        "job_id": job.id,
        "status": job.status,
        "created_at": time.time(),
        **data,
    }
    job.events.append(payload)
    await job.queue.put(payload)
    if job.on_event is not None:
        try:
            await job.on_event(payload)
        except Exception:
            pass
    if job.persist:
        try:
            await persist_job(job)
        except Exception:
            pass


async def stream_job(job: DevDesignerJob):
    for event in job.events:
        yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
    if job.status in {"completed", "failed", "cancelled"}:
        return
    while True:
        try:
            item = await asyncio.wait_for(job.queue.get(), timeout=15)
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"
            continue
        if item is None:
            break
        yield "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"


def _image_from_bytes(data: bytes) -> Image.Image:
    return normalize_user_image(data).convert("RGB")


def _png_bytes(pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _compact(text: str) -> str:
    text = text or ""
    if len(text) <= MAX_CONTEXT_CHARS:
        return text
    return (
        text[: MAX_CONTEXT_CHARS // 2]
        + "\n\n[Context compacted: middle omitted to stay within dev agent budget.]\n\n"
        + text[-MAX_CONTEXT_CHARS // 2 :]
    )


def get_system_prompt_sections(
    section_overrides: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """Structured system prompt sections for the dev dashboard."""
    prompt_loader.ensure_prompts_loaded()
    overrides = section_overrides or {}
    sections = [
        {
            "id": "header",
            "label": "Agent loop (persistence + stop criteria)",
            "path": None,
            "content": DEV_AGENT_LOOP_SECTION,
        },
        {
            "id": "image_gen",
            "label": "Dev image generation (multi-image only)",
            "path": None,
            "content": DEV_IMAGE_GEN_SECTION,
        },
        {
            "id": "designer_prompt_writer",
            "label": "Designer prompt writer system",
            "path": "app/prompts/designer-prompt-writer.md",
            "content": prompt_loader.DESIGNER_PROMPT_WRITER_SYSTEM,
        },
        {
            "id": "catalog_context",
            "label": "Ducon catalog context (no studio wizard tools)",
            "path": None,
            "content": DEV_CATALOG_CONTEXT_SECTION,
        },
    ]
    for section in sections:
        if section["id"] in overrides:
            section["content"] = overrides[section["id"]]
    return sections


def _system_prompt(
    override: Optional[str],
    *,
    mode: str = "append",
    section_overrides: Optional[dict[str, str]] = None,
) -> str:
    if mode == "replace" and override:
        return _compact(override)

    use_overrides = section_overrides if mode == "compose" else None
    parts = [
        f"[{section['label']}]\n{section['content']}"
        for section in get_system_prompt_sections(use_overrides)
    ]
    if mode in {"append", "compose"} and override:
        parts.append("[Dev override]\n" + override)
    return _compact("\n\n".join(parts))


def get_designer_config() -> dict[str, Any]:
    """Dashboard metadata: tools, filesystem contract, default system prompt."""
    sections = get_system_prompt_sections()
    return {
        "tools": list(DESIGNER_TOOL_CATALOG),
        "default_tool_access": list(DEFAULT_DESIGNER_TOOL_ACCESS),
        "tool_descriptions": {
            "ai_search": "Semantic catalog search via ChromaDB embeddings. Agent-callable, one query per call.",
            "keyword_search": "Exact/filter catalog lookup via keyword_search_catalog. Agent-callable, one query per call.",
            "get_image": "Load catalog images into the agent's context so it can see them before referencing.",
            "generate_multi_image": (
                "Fast image generation: one Gemini call with the prompt you write. "
                "image_refs: 'user_photo', 'catalog:<id>', 'generation:<n>'."
            ),
            "generate_multi_image_pipeline": (
                "Full workflow: ImageGenAgent prompt writing, pre-gen verify, "
                "generate → eval → retry loop inside one tool call. Same image_refs."
            ),
            "filesystem": (
                "Scoped file tools the agent can call itself (fs_list, fs_read_text, "
                "fs_write_text, fs_mkdir, fs_delete) under filesystem_root. Nothing is "
                "written automatically — the agent decides."
            ),
        },
        "filesystem": {
            "implementation": "ScopedFilesystemTool (in-house, Python stdlib only — not ripgrep/find)",
            "operations": FILESYSTEM_OPERATIONS,
            "notes": [
                "Paths must stay inside the configured root (no .., no absolute paths).",
                "Exposed to the agent as fs_* tools; the agent chooses what to read/write.",
                "No grep/find/subprocess shell — list, stat, read_text, write_text, mkdir, delete only.",
            ],
        },
        "generation": {
            "default_max_generation_rounds": DEFAULT_MAX_GENERATION_ROUNDS,
            "default_max_turns": DEFAULT_MAX_TURNS,
            "default_aspect_ratio": DEFAULT_ASPECT_RATIO,
            "aspect_ratio_options": ["auto", "1:1", "4:3", "3:4", "16:9", "9:16"],
            "default_max_eval_rounds": DEFAULT_PIPELINE_MAX_EVAL_ROUNDS,
            "default_max_prompt_verify_rounds": DEFAULT_PIPELINE_PROMPT_VERIFY_ROUNDS,
            "wall_clock_budget_s": WALL_CLOCK_BUDGET_S,
            "unlimited_sentinel": 0,
            "loop": "agentic tool-calling loop; the model decides which tools to call and when to finish",
            "stops": [
                "agent_called_finish",
                "agent_stopped_calling_tools",
                "max_turns_reached (skipped when max_turns=0)",
                "max_generations_reached (generation tool disabled, agent told to finish)",
                "wall_clock_budget (skipped when wall_clock_budget_s=0)",
                "error",
            ],
        },
        "system_prompt": {
            "sections": sections,
            "composed_default": _system_prompt(None),
            "max_context_chars": MAX_CONTEXT_CHARS,
        },
        "session": session_config_defaults(),
    }


async def _summarize_messages_for_session(
    messages: list[dict[str, Any]],
    instructions: str,
    *,
    model_pair: ModelRouterPair,
) -> str:
    """LLM summary of middle agent history when client context budget is exceeded."""
    lines: list[str] = []
    for msg in messages[1:]:
        role = msg.get("role")
        if role == "assistant":
            if msg.get("text"):
                lines.append(f"Assistant: {str(msg['text'])[:2000]}")
            for tc in msg.get("tool_calls") or []:
                lines.append(
                    f"Tool call: {tc.get('name')} "
                    f"{json.dumps(tc.get('args') or {}, ensure_ascii=False)[:800]}"
                )
        elif role == "tool":
            raw = msg.get("result")
            snippet = (
                json.dumps(raw, ensure_ascii=False)[:1500]
                if isinstance(raw, dict) else str(raw)[:1500]
            )
            lines.append(f"Tool {msg.get('name')}: {snippet}")
        elif role == "user" and msg.get("text"):
            lines.append(f"User: {str(msg['text'])[:1200]}")
    prompt = (
        f"{instructions.strip()}\n\n--- Conversation to summarize ---\n"
        + "\n".join(lines[-100:])
    )
    return await provider_clients.complete_text(
        pair=model_pair,
        system="You compress agent transcripts for context management.",
        prompt=prompt,
        thinking="disabled",
        max_tokens=4096,
    )


async def _evaluate_generation(
    *,
    pair: ModelRouterPair,
    user_image: Image.Image,
    generated_image: Image.Image,
    generation_prompt: str,
    round_num: int,
    thinking: Optional[str],
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Per-round self-evaluation of a generated Ducon candidate.

    Uses the planning model + a short rubric and returns a normalized dict:
    ``{"approved": bool, "reasons": [str], "defects": [str]}``. On any model
    error the dict carries an ``error`` key so the caller can stop the loop
    while keeping the current output (no retry on eval failure).
    """
    eval_prompt = f"""Evaluate this Ducon redesign candidate.

Image 1: the original client space photo.
Image 2: the AI-generated design candidate (round {round_num}).

Design intent / generation prompt:
{generation_prompt or "No explicit generation prompt."}

Reply with strict JSON: {{"approved": bool, "reasons": [str], "defects": [str]}}""".strip()
    try:
        result = await provider_clients.complete_json(
            pair=pair,
            system=_DEV_DESIGNER_EVAL_SYSTEM,
            prompt=eval_prompt,
            images=[user_image, generated_image],
            thinking="disabled",
            max_tokens=max_tokens,
        )
    except Exception as exc:
        return {
            "approved": False,
            "reasons": [f"Evaluation model failed: {exc}"],
            "defects": [],
            "error": str(exc),
        }

    approved = bool(result.get("approved"))
    reasons = result.get("reasons") or []
    defects = result.get("defects") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    if not isinstance(defects, list):
        defects = [str(defects)]
    return {
        "approved": approved,
        "reasons": [str(r) for r in reasons],
        "defects": [str(d) for d in defects],
    }


# ── Agent-callable tool implementations ──────────────────────────────────────


async def _single_keyword_search(
    *,
    query: str,
    level: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[list[str]] = None,
    limit: int = SEARCH_LIMIT,
) -> list[dict[str, Any]]:
    async with async_session_maker() as db:
        result = await keyword_search_catalog(
            db,
            query=query,
            level=level,
            class_=None,
            category=category,
            tags=tags,
            tag_logic="OR",
            cross_tab=False,
            limit=limit,
        )
    hits: list[dict[str, Any]] = []
    for hit in result.get("hits") or []:
        hits.append(
            {
                "id": int(hit["id"]),
                "name": hit.get("name") or hit.get("filename"),
                "filename": hit.get("filename"),
                "url": hit.get("url"),
                "source": "keyword_search",
                "query": query,
            }
        )
    return hits


async def _single_ai_search(*, query: str, limit: int = SEARCH_LIMIT) -> list[dict[str, Any]]:
    embedding_model = GeminiEmbeddingModel()
    collection = chromadb.get_db_collection()
    embedding = await asyncio.to_thread(embedding_model.get_text_embedding, query)
    result = await asyncio.to_thread(chromadb.retrieve, collection, embedding, limit)
    ids = (result.get("ids") or [[]])[0]
    if not ids:
        return []
    hits: list[dict[str, Any]] = []
    async with async_session_maker() as db:
        candidate_filenames = _filename_candidates(ids)
        rows = (
            await db.execute(select(DBImage).where(DBImage.filename.in_(candidate_filenames)))
        ).scalars().all()
        row_by_filename: dict[str, DBImage] = {}
        for row in rows:
            for candidate in _single_filename_candidates(row.filename):
                row_by_filename[candidate] = row
            row_by_filename[row.filename] = row
        seen: set[int] = set()
        for filename in ids:
            row = row_by_filename.get(filename)
            if not row:
                continue
            cid = int(row.id)
            if cid in seen:
                continue
            seen.add(cid)
            metadata = get_image_metadata(row.filename) or {}
            hits.append(
                {
                    "id": cid,
                    "name": row.name or row.filename,
                    "filename": row.filename,
                    "url": row.url,
                    "metadata": {k: metadata[k] for k in list(metadata)[:8]} if isinstance(metadata, dict) else {},
                    "source": "ai_search",
                    "query": query,
                }
            )
    return hits


async def _load_catalog_images_by_id(ids: list[int]) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    infos: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    async with async_session_maker() as db:
        rows = (await db.execute(select(DBImage).where(DBImage.id.in_(ids)))).scalars().all()
        row_by_id = {int(row.id): row for row in rows}
        for cid in ids:
            row = row_by_id.get(int(cid))
            if row is None:
                infos.append({"id": int(cid), "loaded": False, "error": "not found"})
                continue
            try:
                images.append(await load_ducon_image(row))
                infos.append({"id": int(cid), "loaded": True, "name": row.name or row.filename})
            except Exception as exc:
                infos.append({"id": int(cid), "loaded": False, "error": str(exc)})
    return infos, images


class _AgentToolbox:
    """Executes the designer agent's tool calls and tracks loop state."""

    def __init__(
        self,
        *,
    job: DevDesignerJob,
        user_image: Image.Image,
        tool_access: set[str],
        filesystem_root: Optional[str],
        model_pair: ModelRouterPair,
        image_pair: ModelRouterPair,
        thinking: Optional[str],
        image_thinking: Optional[str],
        aspect_ratio: Optional[str],
        out_rg: str,
        out_pid: str,
        max_generations: int,
        session_config: Optional[Any] = None,
    ) -> None:
        self.job = job
        self.user_image = user_image
        self.tool_access = tool_access
        self.model_pair = model_pair
        self.image_pair = image_pair
        self.thinking = thinking
        self.image_thinking = image_thinking
        self.aspect_ratio = (aspect_ratio or DEFAULT_ASPECT_RATIO).strip() or DEFAULT_ASPECT_RATIO
        self.out_rg = out_rg
        self.out_pid = out_pid
        self.max_generations = max(1, int(max_generations or 1))
        cfg = session_config or parse_session_config({})
        self.session_config = cfg
        self.max_eval_rounds = cfg.max_eval_rounds
        self.max_prompt_verify_rounds = cfg.max_prompt_verify_rounds

        self.generations: list[dict[str, Any]] = []  # {"index", "urls", "pil", "prompt", ...}
        self.catalog_images: dict[int, Image.Image] = {}
        self.sources: dict[int, dict[str, Any]] = {}
        self.eval_history: list[dict[str, Any]] = []
        self.finish_args: Optional[dict[str, Any]] = None
        self.output_idx = 0
        self.fs: Optional[ScopedFilesystemTool] = None
        self.filesystem_root = resolve_job_filesystem_root(filesystem_root, job.id)
        if "filesystem" in tool_access and self.filesystem_root:
            try:
                self.fs = ScopedFilesystemTool(self.filesystem_root)
            except (FilesystemScopeError, OSError, ValueError):
                self.fs = None

    def _generation_tools_enabled(self) -> bool:
        return (
            "generate_multi_image" in self.tool_access
            or "generate_multi_image_pipeline" in self.tool_access
        )

    def _generation_tool_schema(self, *, pipeline: bool) -> dict[str, Any]:
        name = "generate_multi_image_pipeline" if pipeline else "generate_multi_image"
        if pipeline:
            description = (
                "Full image workflow in one call: ImageGenAgent writes the prompt from your "
                "brief, pre-generation verify, then generate → eval → retry until approved or "
                "rounds exhausted. image_refs: 'user_photo' first, then 'catalog:<id>' and/or "
                "'generation:<n>'. The final image is attached to the tool result."
            )
            prompt_desc = "Design brief / hint — the pipeline expands this into a full prompt."
        else:
            description = (
                "Fast image generation: one Gemini call with the prompt you write. "
                "image_refs: 'user_photo' (ALWAYS first), then 'catalog:<id>' references, and/or "
                "'generation:<n>' to refine a previous candidate. The generated image is "
                "attached to the tool result so you can judge it."
            )
            prompt_desc = "Full generation prompt."
        return {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": prompt_desc},
                    "image_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "'user_photo', 'catalog:<id>', 'generation:<n>'.",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "description": (
                            "Optional override. Default job setting is 'auto' (match client photo). "
                            "Or: '1:1', '4:3', '3:4', '16:9', '9:16'."
                        ),
                    },
                },
                "required": ["prompt", "image_refs"],
            },
        }

    def declarations(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        if "ai_search" in self.tool_access:
            tools.append({
                "name": "ai_search",
                "description": (
                    "Semantic search over the Ducon catalog (embeddings). One natural-language "
                    "query per call, e.g. 'grey porcelain paving modern patio'. Returns catalog "
                    "hits with ids. Run as many searches as you need."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural-language search query."},
                        "limit": {"type": "integer", "description": "Max hits (default 5, max 8)."},
                    },
                    "required": ["query"],
                },
            })
        if "keyword_search" in self.tool_access:
            tools.append({
                "name": "keyword_search",
                "description": (
                    "Exact/keyword catalog lookup with optional filters. Good for product names, "
                    "levels, categories, or tags you already know."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "level": {"type": "string", "description": "Optional catalog level filter."},
                        "category": {"type": "string", "description": "Optional category filter."},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            })
        if "get_image" in self.tool_access:
            tools.append({
                "name": "get_image",
                "description": (
                    "Load catalog images into your context so you can SEE them (attached to the "
                    f"tool result). Up to {MAX_IMAGES_PER_GET} ids per call. Always inspect a "
                    "reference before using it in generate_multi_image."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_ids": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["image_ids"],
                },
            })
        if "generate_multi_image" in self.tool_access:
            tools.append(self._generation_tool_schema(pipeline=False))
        if "generate_multi_image_pipeline" in self.tool_access:
            tools.append(self._generation_tool_schema(pipeline=True))
        if self._generation_tools_enabled():
            tools.append({
                "name": "evaluate_design",
                "description": (
                    "Independent quality-gate evaluation of one of your generated candidates "
                    "against the client photo. Returns approved/reasons/defects. Use it to decide "
                    "whether to refine or finish. (Optional when using generate_multi_image_pipeline, "
                    "which runs its own inner eval loop.)"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "generation_index": {
                            "type": "integer",
                            "description": "1-based index from a generation tool (first is 1, not 0).",
                        },
                    },
                    "required": ["generation_index"],
                },
            })
        if "filesystem" in self.tool_access:
            tools.extend([
                {
                    "name": "fs_list",
                    "description": "List files/dirs under your scratch directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Relative path (default '.')."}},
                    },
                },
                {
                    "name": "fs_read_text",
                    "description": "Read a text file from your scratch directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
                {
                    "name": "fs_write_text",
                    "description": "Write (or append to) a text file in your scratch directory — notes, plans, briefs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "append": {"type": "boolean"},
                        },
                        "required": ["path", "content"],
                    },
                },
                {
                    "name": "fs_mkdir",
                    "description": "Create a directory in your scratch directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
                {
                    "name": "fs_delete",
                    "description": "Delete a file or directory in your scratch directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "recursive": {"type": "boolean"},
                        },
                        "required": ["path"],
                    },
                },
            ])
        tools.append({
            "name": "finish",
            "description": (
                "End the job when you have a design you would present to a client (or cannot "
                "proceed). Select the best generation and summarize what you did and why."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "What you designed, key choices, sources used."},
                    "selected_generation": {"type": "integer", "description": "Index of the generation you are selecting."},
                },
                "required": ["summary"],
            },
        })
        return tools

    def _budget_note(self) -> dict[str, Any]:
        return {
            "generations_used": len(self.generations),
            "generations_remaining": max(0, self.max_generations - len(self.generations)),
        }

    def _resolve_refs(self, refs: list[Any]) -> tuple[list[ImageDescriptor], list[str]]:
        descriptors: list[ImageDescriptor] = []
        errors: list[str] = []
        for raw in refs:
            ref = str(raw or "").strip()
            low = ref.lower()
            if low in {"user_photo", "user photo", "client space photo", "user_image"}:
                descriptors.append(
                    ImageDescriptor(label="client space photo", type="file", pil_image=self.user_image)
                )
            elif low.startswith("catalog:"):
                try:
                    cid = int(low.split(":", 1)[1])
                except ValueError:
                    errors.append(f"invalid catalog ref: {ref}")
                    continue
                src = self.sources.get(cid) or {}
                descriptors.append(
                    ImageDescriptor(
                        label=str(src.get("name") or f"catalog {cid}"),
                        type="catalog_id",
                        source=str(cid),
                    )
                )
            elif low.startswith("generation:"):
                try:
                    gidx = int(low.split(":", 1)[1])
                except ValueError:
                    errors.append(f"invalid generation ref: {ref}")
                    continue
                gen = next((g for g in self.generations if g["index"] == gidx), None)
                if gen is None or gen.get("pil") is None:
                    errors.append(f"unknown generation index: {gidx}")
                    continue
                descriptors.append(
                    ImageDescriptor(
                        label="previous design candidate",
                        type="file",
                        pil_image=gen["pil"],
                    )
                )
            elif low.isdigit():
                descriptors.append(
                    ImageDescriptor(label=f"catalog {low}", type="catalog_id", source=low)
                )
            else:
                errors.append(f"unknown image ref: {ref}")
        # The pipeline expects the client photo first; prepend if the agent forgot.
        if not any(d.label == "client space photo" for d in descriptors):
            descriptors.insert(
                0, ImageDescriptor(label="client space photo", type="file", pil_image=self.user_image)
            )
        return descriptors, errors

    async def _run_generation(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        use_pipeline: bool,
    ) -> tuple[dict[str, Any], list[Image.Image], Optional[str]]:
        tool_key = "generate_multi_image_pipeline" if use_pipeline else "generate_multi_image"
        if tool_key not in self.tool_access:
            return {"error": f"{tool_key} is disabled for this job"}, [], None
        if len(self.generations) >= self.max_generations:
            return {
                "error": "generation budget exhausted",
                **self._budget_note(),
                "hint": "Call finish with your best generation.",
            }, [], None
        prompt = str(args.get("prompt") or "").strip()
        refs = args.get("image_refs") or ["user_photo"]
        if not prompt:
            return {"error": "prompt is required"}, [], None
        descriptors, ref_errors = self._resolve_refs(refs if isinstance(refs, list) else [refs])
        if ref_errors:
            if len(descriptors) <= 1:
                return {
                    "error": "could not resolve image_refs",
                    "details": ref_errors,
                    "hint": "Use user_photo, catalog:<id> from search hits, or generation:<n> (1-based).",
                }, [], None
        aspect = str(args.get("aspect_ratio") or "").strip() or self.aspect_ratio
        gen_index = len(self.generations) + 1
        mode_label = "pipeline" if use_pipeline else "direct"
        await emit(
            self.job,
            "generation_started",
            message=f"Generating candidate {gen_index} ({mode_label}).",
            round=gen_index,
            pipeline=use_pipeline,
        )

        async def _on_step(step: Any) -> None:
            await emit(
                self.job,
                "pipeline_step",
                round=gen_index,
                kind=step.kind,
                status=step.status,
                model=step.model,
                thinking=step.thinking,
                prompt_used=step.prompt_used,
                error=step.error,
                duration_ms=step.duration_ms,
            )

        async with async_session_maker() as db:
            result = await generate_multi_image(
                user_id=None,
                guest_session_id=None,
                prompt=prompt,
                descriptors=descriptors,
                model="pro",
                aspect_ratio=aspect,
                db=db,
                enable_verify=use_pipeline,
                dry_run=True,
                image_model=self.image_pair.model_id,
                image_thinking=self.image_thinking,
                prompt_model=self.model_pair.model_id,
                prompt_thinking=self.thinking,
                max_eval_rounds=self.max_eval_rounds,
                max_prompt_verify_rounds=self.max_prompt_verify_rounds,
                on_step=_on_step if use_pipeline else None,
            )
        pil_images = result.get("pil_images") or []
        if not pil_images:
            return {"error": "generation produced no images", **self._budget_note()}, [], None
        urls: list[dict[str, Any]] = []
        for img in pil_images:
            url = await store.save_output_image(self.out_rg, self.out_pid, self.output_idx, _png_bytes(img))
            urls.append({"index": self.output_idx, "round": gen_index, "url": url})
            self.output_idx += 1
        stored_prompt = str(result.get("final_prompt") or prompt)
        self.generations.append(
            {
                "index": gen_index,
                "urls": urls,
                "pil": pil_images[0],
                "prompt": stored_prompt,
                "pipeline": use_pipeline,
                "approved": result.get("approved") if use_pipeline else None,
                "retries": result.get("retries") if use_pipeline else None,
            }
        )
        await emit(self.job, "generation_done", round=gen_index, output_images=urls, pipeline=use_pipeline)
        payload: dict[str, Any] = {
            "generation_index": gen_index,
            "output_images": urls,
            "resolved_image_refs": refs,
            "model_used": result.get("model_used") or self.image_pair.model_id,
            "mode": mode_label,
            **self._budget_note(),
            "note": (
                f"Attached: generated design candidate generation_index={gen_index} "
                "(1-based). Compare to the client photo before refining or finishing."
            ),
        }
        if use_pipeline:
            payload.update(
                {
                    "final_prompt": stored_prompt,
                    "approved": result.get("approved"),
                    "retries": result.get("retries"),
                    "pipeline_steps": len(result.get("steps") or []),
                }
            )
        if ref_errors:
            payload["image_ref_warnings"] = ref_errors
        caption = (
            f"Attached: generated design candidate generation_index={gen_index}. "
            "Image 1 in this result is your new design — compare it to the client photo."
        )
        return payload, [pil_images[0]], caption

    async def execute(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], list[Image.Image], Optional[str]]:
        """Run one tool call. Returns (json_safe_result, images_for_model, image_caption)."""
        try:
            return await self._execute_inner(name, args)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}, [], None

    async def _execute_inner(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], list[Image.Image], Optional[str]]:
        if name == "ai_search":
            if "ai_search" not in self.tool_access:
                return {"error": "ai_search is disabled for this job"}, [], None
            query = str(args.get("query") or "").strip()
            if not query:
                return {"error": "query is required"}, [], None
            limit = min(max(int(args.get("limit") or SEARCH_LIMIT), 1), 8)
            hits = await _single_ai_search(query=query, limit=limit)
            for h in hits:
                self.sources[h["id"]] = h
            return {"query": query, "hits": hits, "hint": "Use get_image to inspect promising ids."}, [], None

        if name == "keyword_search":
            if "keyword_search" not in self.tool_access:
                return {"error": "keyword_search is disabled for this job"}, [], None
            query = str(args.get("query") or "").strip()
            if not query:
                return {"error": "query is required"}, [], None
            tags = args.get("tags")
            hits = await _single_keyword_search(
                query=query,
                level=(str(args["level"]) if args.get("level") else None),
                category=(str(args["category"]) if args.get("category") else None),
                tags=[str(t) for t in tags] if isinstance(tags, list) else None,
                limit=min(max(int(args.get("limit") or SEARCH_LIMIT), 1), 8),
            )
            for h in hits:
                self.sources[h["id"]] = h
            return {"query": query, "hits": hits}, [], None

        if name == "get_image":
            if "get_image" not in self.tool_access:
                return {"error": "get_image is disabled for this job"}, [], None
            raw_ids = args.get("image_ids") or []
            if not isinstance(raw_ids, list) or not raw_ids:
                return {"error": "image_ids (non-empty array) is required"}, [], None
            ids = [int(i) for i in raw_ids][:MAX_IMAGES_PER_GET]
            infos, images = await _load_catalog_images_by_id(ids)
            for cid, img in zip([i["id"] for i in infos if i.get("loaded")], images):
                self.catalog_images[cid] = img
            loaded = [i for i in infos if i.get("loaded")]
            caption = (
                "Attached catalog reference images (in order): "
                + ", ".join(f"id {i['id']} ({i.get('name') or 'catalog'})" for i in loaded)
                if loaded
                else None
            )
            return (
                {
                    "images": infos,
                    "note": "Loaded images are attached to this result — use catalog:<id> in image_refs.",
                },
                images,
                caption,
            )

        if name == "generate_multi_image":
            return await self._run_generation(tool_name=name, args=args, use_pipeline=False)

        if name == "generate_multi_image_pipeline":
            return await self._run_generation(tool_name=name, args=args, use_pipeline=True)

        if name == "evaluate_design":
            if not self.generations:
                return {"error": "no generations to evaluate yet"}, [], None
            gidx = int(args.get("generation_index") or 0)
            gen = next((g for g in self.generations if g["index"] == gidx), None)
            if gen is None:
                # Tolerate off-by-one/missing index: evaluate the latest candidate.
                gen = self.generations[-1]
                gidx = gen["index"]
            eval_result = await _evaluate_generation(
                pair=self.model_pair,
                user_image=self.user_image,
                generated_image=gen["pil"],
                generation_prompt=str(gen.get("prompt") or ""),
                round_num=gidx,
                thinking=self.thinking,
                max_tokens=self.session_config.eval_max_tokens,
            )
            entry = {"round": gidx, **eval_result}
            self.eval_history.append(entry)
            await emit(
                self.job, "eval",
                round=gidx,
                approved=eval_result.get("approved"),
                reasons=eval_result.get("reasons"),
                defects=eval_result.get("defects"),
                error=eval_result.get("error"),
            )
            return entry, [], None

        if name.startswith("fs_"):
            if "filesystem" not in self.tool_access:
                return {"error": "filesystem tools are disabled for this job"}, [], None
            if self.fs is None:
                return {"error": "no filesystem_root configured — set 'Scoped filesystem directory' in the job form"}, [], None
            op = name[3:]
            if op not in FILESYSTEM_OPERATIONS and op != "list":
                return {"error": f"unsupported filesystem op: {op}"}, [], None
            try:
                result = await asyncio.to_thread(self.fs.call, op, **args)
                return result, [], None
            except (FilesystemScopeError, OSError, ValueError, FileNotFoundError, NotADirectoryError) as exc:
                return {"error": str(exc)}, [], None

        if name == "finish":
            self.finish_args = {
                "summary": str(args.get("summary") or ""),
                "selected_generation": args.get("selected_generation"),
            }
            return {"ok": True, "note": "Job will end after this turn."}, [], None

        return {"error": f"unknown tool: {name}"}, [], None


async def run_dev_designer_job(
    *,
    job: DevDesignerJob,
    user_image_bytes: bytes,
    user_prompt: str,
    model_pair: ModelRouterPair,
    image_pair: ModelRouterPair,
    thinking: Optional[str],
    image_thinking: Optional[str],
    system_prompt: Optional[str],
    system_prompt_mode: str = "append",
    system_prompt_sections: Optional[dict[str, str]] = None,
    tool_access: set[str],
    filesystem_root: Optional[str],
    aspect_ratio: Optional[str],
    output_run_group_id: Optional[str] = None,
    output_process_id: str = "designer",
    max_generation_rounds: int = DEFAULT_MAX_GENERATION_ROUNDS,
    max_turns: int = DEFAULT_MAX_TURNS,
    wall_clock_budget_s: Optional[int] = None,
    persist: bool = True,
) -> None:
    """Agentic tool-calling loop (Claude Code / Cursor style).

    The model owns the loop: it decides which tools to call (searches, image
    loads, generation, evaluation, filesystem, finish) and when to stop.
    The harness enforces optional budgets: max model turns (0 = unlimited),
    max generations, and wall-clock cap (0 = unlimited).
    """
    out_rg = output_run_group_id or job.id
    started_monotonic = time.monotonic()
    try:
        job.status = "running"
        if persist:
            await persist_job(job)
        user_image = _image_from_bytes(user_image_bytes)
        input_meta = job.input or {}
        session_cfg = parse_session_config(input_meta)
        await emit(
            job,
            "input_image",
            url=input_meta.get("upload_url"),
            name=input_meta.get("upload_name"),
            prompt=user_prompt,
        )
        system = _system_prompt(
            system_prompt,
            mode=system_prompt_mode,
            section_overrides=system_prompt_sections,
        )
        await emit(job, "system_prompt_used", mode=system_prompt_mode, prompt=system)

        toolbox = _AgentToolbox(
            job=job,
            user_image=user_image,
            tool_access=tool_access,
            filesystem_root=filesystem_root,
            model_pair=model_pair,
            image_pair=image_pair,
            thinking=thinking,
            image_thinking=image_thinking,
            aspect_ratio=aspect_ratio,
            out_rg=out_rg,
            out_pid=output_process_id,
            max_generations=max_generation_rounds,
            session_config=session_cfg,
        )
        tools = toolbox.declarations()
        turn_cap = _resolve_max_turns(max_turns)
        wall_cap_s = _resolve_wall_clock_budget_s(wall_clock_budget_s)

        turns_label = "unlimited (stop via finish or when you stop calling tools)" if turn_cap is None else str(turn_cap)
        wall_label = "unlimited" if wall_cap_s is None else f"{wall_cap_s // 60} minutes"

        task_text = (
            f"Client brief: {user_prompt.strip() or '(no brief supplied — design a premium Ducon upgrade for this space)'}\n"
            f"Preferred aspect ratio: {toolbox.aspect_ratio} "
            f"({'match client photo' if toolbox.aspect_ratio == 'auto' else 'fixed ratio'})\n"
            f"Budgets: {turns_label} model turns, {toolbox.max_generations} generations, "
            f"{wall_label} wall clock.\n"
            "The attached image is the client's space photo (reference it as \"user_photo\").\n"
            "Work autonomously with your tools and call finish when done."
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "text": task_text, "images": [user_image]}
        ]

        stop_reason = "max_turns_reached"
        turns_used = 0
        nudged = False
        consecutive_model_errors = 0
        turn = 0

        while True:
            turn += 1
            turns_used = turn
            if job.cancel_requested:
                stop_reason = "cancelled"
                break
            if turn_cap is not None and turn > turn_cap:
                stop_reason = "max_turns_reached"
                break
            if wall_cap_s is not None and time.monotonic() - started_monotonic > wall_cap_s:
                stop_reason = "wall_clock_budget"
                break

            async def _summarize(msgs: list[dict[str, Any]], instructions: str) -> str:
                return await _summarize_messages_for_session(
                    msgs, instructions, model_pair=model_pair,
                )

            compacted_messages, compact_meta = await maybe_compact_messages(
                messages,
                config=session_cfg,
                system=system,
                router=model_pair.router,
                summarize=_summarize if session_cfg.effective_summarizer_instructions() else None,
            )
            if compact_meta.get("actions"):
                await emit(job, "session_compacted", turn=turn, **compact_meta)
            messages = compacted_messages

            try:
                resp = await provider_clients.chat_with_tools(
                    pair=model_pair,
                    system=system,
                    messages=messages,
                    tools=tools,
                    thinking=thinking,
                    max_tokens=session_cfg.max_tokens,
                    session_config=session_cfg,
                )
            except Exception as exc:
                consecutive_model_errors += 1
                await emit(job, "tool_error", tool="model", error=str(exc), turn=turn)
                if consecutive_model_errors >= 2:
                    stop_reason = f"model_error: {exc}"
                    break
                await asyncio.sleep(2)
                continue
            consecutive_model_errors = 0

            finish_reason = str(resp.get("finish_reason") or "").lower()
            if finish_reason in {"max_tokens", "length"} or "max_token" in finish_reason:
                await emit(
                    job,
                    "status",
                    turn=turn,
                    message=(
                        "Model output may be truncated (hit max_tokens). "
                        "If tool calls look incomplete, the next turn will continue with compacted context."
                    ),
                    finish_reason=resp.get("finish_reason"),
                )

            text = str(resp.get("text") or "").strip()
            tool_calls = resp.get("tool_calls") or []
            if text:
                await emit(job, "assistant_message", text=text, turn=turn)
            messages.append({
                "role": "assistant",
                "text": text or None,
                "tool_calls": tool_calls,
                # Raw provider content (thought signatures etc.) replayed verbatim
                # on the next turn — required by Gemini 3 function calling.
                "provider_state": resp.get("provider_state"),
            })

            if not tool_calls:
                # Model stopped calling tools. If it produced nothing yet, nudge
                # once (persistence reminder); otherwise respect its decision.
                if not toolbox.generations and not nudged:
                    nudged = True
                    messages.append({
                        "role": "user",
                        "text": (
                            "You have not produced a design yet. Continue working: search the "
                            "catalog, inspect references, and call generate_multi_image. If you "
                            "truly cannot proceed, call finish and explain why."
                        ),
                    })
                    continue
                stop_reason = "agent_stopped_calling_tools"
                break

            finished = False
            for tc in tool_calls:
                await emit(
                    job, "tool_call",
                    turn=turn,
                    id=tc.get("id"),
                    name=tc.get("name"),
                    args=tc.get("args") or {},
                )
                result, images, image_caption = await toolbox.execute(str(tc.get("name") or ""), tc.get("args") or {})
                await emit(
                    job, "tool_result",
                    turn=turn,
                    id=tc.get("id"),
                    name=tc.get("name"),
                    result=result,
                    output_images=(result.get("output_images") if isinstance(result, dict) else None),
                    image_count=len(images),
                )
                append_tool_message(
                    messages,
                    tool_call_id=tc.get("id"),
                    name=str(tc.get("name") or ""),
                    result=result,
                    images=images,
                    image_caption=image_caption,
                    config=session_cfg,
                )
                if tc.get("name") == "finish":
                    finished = True
            if finished:
                stop_reason = "agent_called_finish"
                break

        # ── Assemble final result from what the agent actually did ────────────
        selected_idx = None
        summary = ""
        if toolbox.finish_args:
            summary = toolbox.finish_args.get("summary") or ""
            raw_sel = toolbox.finish_args.get("selected_generation")
            if raw_sel is not None:
                try:
                    selected_idx = int(raw_sel)
                except (TypeError, ValueError):
                    selected_idx = None
        selected_gen = next(
            (g for g in toolbox.generations if g["index"] == selected_idx),
            toolbox.generations[-1] if toolbox.generations else None,
        )
        elapsed_s = round(time.monotonic() - started_monotonic, 1)

        if stop_reason == "cancelled":
            job.status = "cancelled"
            job.error = "Cancelled by user"
            await emit(job, "cancelled", message="Job cancelled by user.")
            return

        final = {
            "job_id": job.id,
            "design_generation": {
                "output_images": (selected_gen or {}).get("urls") or [],
                "rounds": [
                    {"round": g["index"], "output_images": g["urls"]} for g in toolbox.generations
                ],
                "eval_history": toolbox.eval_history,
                "rounds_run": len(toolbox.generations),
                "max_generation_rounds": toolbox.max_generations,
                "selected_generation": (selected_gen or {}).get("index"),
                "model_used": image_pair.model_id,
                "approved": (toolbox.eval_history[-1].get("approved") if toolbox.eval_history else None),
            },
            "design_plan": {
                "summary": summary,
                "final_generation_prompt": (selected_gen or {}).get("prompt"),
            },
            "sources_used": list(toolbox.sources.values()),
            "elements_used": [],
            "metadata": {
                "model_pair": provider_registry.serialize_pair(model_pair),
                "image_pair": provider_registry.serialize_pair(image_pair),
                "thinking": thinking,
                "image_thinking": image_thinking,
                "tool_access": sorted(tool_access),
                "filesystem_root": toolbox.filesystem_root if "filesystem" in tool_access else None,
                "filesystem_base": filesystem_root if "filesystem" in tool_access else None,
                "aspect_ratio": aspect_ratio,
                "session": session_cfg.to_dict(),
                "loop": "agentic",
                "stop_reason": stop_reason,
                "turns_used": turns_used,
                "max_turns": turn_cap,
                "wall_clock_budget_s": wall_cap_s,
                "max_generation_rounds": toolbox.max_generations,
                "elapsed_s": elapsed_s,
            },
        }
        job.status = "completed"
        job.final = final
        await emit(job, "final", **final)
    except asyncio.CancelledError:
        job.status = "cancelled"
        job.error = "Cancelled by user"
        await emit(job, "cancelled", message="Job cancelled by user.")
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        await emit(job, "error", message=str(exc))
    finally:
        if persist:
            await persist_job(job)
        await job.queue.put(None)


async def run_designer_benchmark(
    *,
    process_id: str,
    run_group_id: str,
    user_image_bytes: bytes,
    user_prompt: str,
    model_pair: ModelRouterPair,
    image_pair: ModelRouterPair,
    thinking: Optional[str],
    image_thinking: Optional[str],
    system_prompt: Optional[str],
    system_prompt_mode: str = "append",
    system_prompt_sections: Optional[dict[str, str]] = None,
    tool_access: set[str],
    filesystem_root: Optional[str],
    aspect_ratio: Optional[str],
    max_generation_rounds: int = DEFAULT_MAX_GENERATION_ROUNDS,
    max_turns: int = DEFAULT_MAX_TURNS,
    wall_clock_budget_s: Optional[int] = None,
    on_event: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None,
) -> tuple[Optional[str], list[bytes], dict[str, Any]]:
    """Run the designer pipeline for a benchmark process (no disk job snapshot)."""
    job = DevDesignerJob(
        id=process_id,
        input={"prompt": user_prompt},
        persist=False,
        on_event=on_event,
    )
    await run_dev_designer_job(
        job=job,
        user_image_bytes=user_image_bytes,
        user_prompt=user_prompt,
        model_pair=model_pair,
        image_pair=image_pair,
        thinking=thinking,
        image_thinking=image_thinking,
        system_prompt=system_prompt,
        system_prompt_mode=system_prompt_mode,
        system_prompt_sections=system_prompt_sections,
        tool_access=tool_access,
        filesystem_root=filesystem_root,
        aspect_ratio=aspect_ratio,
        output_run_group_id=run_group_id,
        output_process_id=process_id,
        max_generation_rounds=max_generation_rounds,
        max_turns=max_turns,
        wall_clock_budget_s=wall_clock_budget_s,
        persist=False,
    )

    output_images: list[bytes] = []
    if job.final:
        gen = job.final.get("design_generation") or {}
        rounds = gen.get("rounds") or []
        if rounds:
            round_image_lists = [r.get("output_images") or [] for r in rounds]
        else:
            # Backwards-compat: older finals only carried a flat output_images list.
            round_image_lists = [gen.get("output_images") or []]
        for img_list in round_image_lists:
            for img in img_list:
                url = str(img.get("url") or "")
                prefix = f"/dev/outputs/{run_group_id}/{process_id}/"
                if url.startswith(prefix):
                    idx = url.rsplit("/", 1)[-1].replace(".png", "")
                    path = store.output_path(run_group_id, process_id, int(idx))
                    if path.exists():
                        output_images.append(path.read_bytes())

    final_prompt = None
    if job.final:
        plan = job.final.get("design_plan")
        if isinstance(plan, dict):
            final_prompt = json.dumps(plan, ensure_ascii=False)
        elif plan:
            final_prompt = str(plan)

    return final_prompt, output_images, job.final or {}
