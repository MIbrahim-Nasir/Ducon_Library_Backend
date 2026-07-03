"""
Studio wizard Step 4 — AI search agent that curates 9 design directions.

Uses multimodal Chroma search + visual inspection loops (similar spirit to the designer
job reference search, but oriented toward user-facing direction cards).
"""
from __future__ import annotations

import asyncio
import io
import json
import os
from dataclasses import dataclass, field
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Optional

from google.genai import types
from google.genai.types import Content, FunctionDeclaration, GenerateContentConfig, Part, Tool, ThinkingConfig
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import chromadb, prompt_loader
from app.db.models import Image as DBImage
from app.designer_agent import _filename_candidates, _single_filename_candidates
from app.gemini import get_gemini_client
from app.image_utils import get_image_metadata, load_ducon_image, normalize_user_image
from app.catalog_keyword_search import keyword_search_catalog
from app.search_tools import AI_SEARCH_WHEN, keyword_search_studio_spec
from app.ml import GeminiEmbeddingModel

# ── Studio directions curator (Step 4) — configure via .env ───────────────────
# STUDIO_DIRECTIONS_MODEL          — Gemini model id (default: gemini-3-flash-preview)
# STUDIO_DIRECTIONS_THINKING_LEVEL — none | minimal | low | medium | high (default: High)
from app.admin.settings_store import cfg, cfg_str

STUDIO_DIRECTIONS_MAX_TURNS = 8
STUDIO_DIRECTIONS_MAX_SEARCHES = int(os.getenv("STUDIO_DIRECTIONS_MAX_SEARCHES", "6"))
STUDIO_DIRECTIONS_SEARCH_LIMIT = int(os.getenv("STUDIO_DIRECTIONS_SEARCH_LIMIT", "8"))
STUDIO_DIRECTIONS_INSPECT_LIMIT = int(os.getenv("STUDIO_DIRECTIONS_INSPECT_LIMIT", "8"))
_PRESEARCH_QUERIES = int(os.getenv("STUDIO_DIRECTIONS_PRESEARCH_QUERIES", "3"))
_PRESEARCH_LIMIT = int(os.getenv("STUDIO_DIRECTIONS_PRESEARCH_LIMIT", "10"))
_PREINSPECT_TOP = int(os.getenv("STUDIO_DIRECTIONS_PREINSPECT_TOP", "10"))
_DIRECTION_COUNT = 9


def studio_directions_model() -> str:
    return cfg_str("STUDIO_DIRECTIONS_MODEL", "gemini-3-flash-preview").strip()


def studio_directions_thinking_level() -> str | None:
    """Return thinking level string, or None when thinking is disabled."""
    raw = cfg_str("STUDIO_DIRECTIONS_THINKING_LEVEL", "High").strip()
    if not raw or raw.lower() in ("none", "off", "false", "0", "disabled"):
        return None
    return raw


def _studio_directions_thinking_config() -> ThinkingConfig | None:
    level = studio_directions_thinking_level()
    if level is None:
        return None
    return ThinkingConfig(thinking_level=level)


def _studio_directions_generate_config() -> GenerateContentConfig:
    prompt_loader.ensure_prompts_loaded()
    kwargs: dict[str, Any] = {
        "system_instruction": prompt_loader.STUDIO_DIRECTIONS_AGENT_SYSTEM,
        "tools": [_AGENT_TOOLS],
    }
    thinking = _studio_directions_thinking_config()
    if thinking is not None:
        kwargs["thinking_config"] = thinking
    return GenerateContentConfig(**kwargs)

_SEARCH_TOOL = FunctionDeclaration(
    name="ai_search",
    description=(
        "Semantic AI search of the Ducon catalog (PREFERRED for direction curation). "
        f"{AI_SEARCH_WHEN} "
        "Uses the user's space photo plus your text query to find visually and semantically "
        "relevant designs. Always include space-type and style keywords from the payload."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Detailed Ducon outdoor design search query.",
            },
            "limit": {
                "type": "integer",
                "description": "Max catalog hits to return (default 14).",
            },
        },
        "required": ["query"],
    },
)

_INSPECT_TOOL = FunctionDeclaration(
    name="inspect_designs",
    description=(
        "Load catalog design images so you can visually compare them to the user's space "
        "photo. Call with promising catalog_id values returned from ai_search."
    ),
    parameters={
        "type": "object",
        "properties": {
            "catalog_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Catalog image IDs to inspect (max 8 per call).",
            },
        },
        "required": ["catalog_ids"],
    },
)

_SUBMIT_TOOL = FunctionDeclaration(
    name="submit_directions",
    description="Submit the final 9 design directions for the user to pick from.",
    parameters={
        "type": "object",
        "properties": {
            "directions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "catalog_id": {"type": "integer"},
                        "title": {"type": "string"},
                        "subtitle": {"type": "string"},
                    },
                    "required": ["catalog_id", "title", "subtitle"],
                },
                "description": "Exactly 9 distinct catalog directions.",
            },
        },
        "required": ["directions"],
    },
)

_SHORTLIST_TOOL = FunctionDeclaration(
    name="shortlist_directions",
    description=(
        "Confirm picks so far after visually inspecting them. Call this after every "
        "inspect_designs batch with the designs you are confident about. Can be fewer "
        "than 9 total across all calls — the tool returns total_shortlisted and "
        "remaining_needed so you know how many more to find. The user sees these cards "
        "appear in real time while you continue searching."
    ),
    parameters={
        "type": "object",
        "properties": {
            "directions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "catalog_id": {"type": "integer"},
                        "title": {"type": "string"},
                        "subtitle": {"type": "string"},
                    },
                    "required": ["catalog_id", "title", "subtitle"],
                },
                "description": "Visually confirmed designs to shortlist (subset of the inspected batch).",
            },
        },
        "required": ["directions"],
    },
)

_KEYWORD_TOOL = FunctionDeclaration(**keyword_search_studio_spec())

_AGENT_TOOLS = Tool(function_declarations=[
    _SEARCH_TOOL,
    _KEYWORD_TOOL,
    _INSPECT_TOOL,
    _SHORTLIST_TOOL,
    _SUBMIT_TOOL,
])


def _pil_part(img: Image.Image) -> Part:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def _serialize_catalog_row(row: DBImage, *, query: str = "") -> dict[str, Any]:
    meta = get_image_metadata(row.filename) or {}
    return {
        "catalog_id": int(row.id),
        "name": row.name or row.filename,
        "filename": row.filename,
        "class": getattr(row, "class_", None) or meta.get("class"),
        "theme": getattr(row, "theme", None) or meta.get("theme"),
        "project": getattr(row, "project", None) or meta.get("project"),
        "tags": meta.get("tags") or [],
        "level": meta.get("level"),
        "found_via_query": query,
    }


@dataclass
class StudioDirectionsSession:
    conversation: list[Content] = field(default_factory=list)
    candidate_pool: dict[int, dict[str, Any]] = field(default_factory=dict)
    rejected_ids: set[int] = field(default_factory=set)
    submitted: Optional[list[dict[str, Any]]] = None
    shortlisted: list[dict[str, Any]] = field(default_factory=list)
    search_count: int = 0
    directions_streamed_to_ui: bool = False
    inspect_ids_seen: set[int] = field(default_factory=set)
    pending_visual_parts: list[Part] = field(default_factory=list)


async def _resolve_rows_by_filenames(
    db: AsyncSession,
    filenames: list[str],
) -> dict[str, DBImage]:
    if not filenames:
        return {}
    candidate_filenames = _filename_candidates(filenames)
    rows = (
        await db.execute(select(DBImage).where(DBImage.filename.in_(candidate_filenames)))
    ).scalars().all()
    row_by_filename: dict[str, DBImage] = {}
    for row in rows:
        for candidate in _single_filename_candidates(row.filename):
            row_by_filename[candidate] = row
        row_by_filename[row.filename] = row
    return row_by_filename


async def _run_keyword_search(
    *,
    db: AsyncSession,
    session: StudioDirectionsSession,
    query: str,
    opts: dict[str, Any] | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    opts = opts or {}
    print(f"[StudioDirections] keyword_search: {query!r} opts={opts}")
    result = await keyword_search_catalog(
        db,
        query=str(query or ""),
        level=opts.get("level"),
        class_=opts.get("class"),
        category=opts.get("category"),
        tags=opts.get("tags"),
        tag_logic=str(opts.get("tagLogic") or opts.get("matchMode") or opts.get("tag_logic") or "OR"),
        cross_tab=bool(opts.get("crossTab") or opts.get("cross_tab")),
        limit=limit,
    )
    for record in result.get("hits") or []:
        cid = int(record["id"])
        if cid in session.rejected_ids:
            continue
        session.candidate_pool[cid] = record
    return result


async def _run_ai_search(
    *,
    db: AsyncSession,
    collection,
    embedding_model: GeminiEmbeddingModel,
    session: StudioDirectionsSession,
    query: str,
    limit: int,
    user_image_bytes: bytes,
    user_mime: str,
) -> dict[str, Any]:
    if session.search_count >= STUDIO_DIRECTIONS_MAX_SEARCHES:
        return {
            "error": "Search budget exhausted — use inspect_designs and submit_directions.",
            "hits": [],
        }

    session.search_count += 1
    limit = max(4, min(limit or STUDIO_DIRECTIONS_SEARCH_LIMIT, 20))
    print(f"[StudioDirections] ai_search #{session.search_count}: {query!r} (limit={limit})")

    embedding = await asyncio.to_thread(
        embedding_model.get_multimodal_embedding,
        query,
        user_image_bytes,
        user_mime,
    )
    result = chromadb.retrieve(collection, embedding, limit)
    filenames = (result.get("ids") or [[]])[0]
    row_by_filename = await _resolve_rows_by_filenames(db, filenames)

    hits: list[dict[str, Any]] = []
    for filename in filenames:
        row = row_by_filename.get(filename)
        if not row:
            continue
        cid = int(row.id)
        if cid in session.rejected_ids:
            continue
        record = _serialize_catalog_row(row, query=query)
        session.candidate_pool[cid] = record
        hits.append(record)

    return {
        "query": query,
        "hits": hits,
        "hit_count": len(hits),
        "search_number": session.search_count,
        "unique_candidates_in_pool": len(session.candidate_pool),
    }


async def _run_inspect_designs(
    *,
    db: AsyncSession,
    session: StudioDirectionsSession,
    catalog_ids: list[int],
) -> dict[str, Any]:
    ids = []
    for raw in catalog_ids or []:
        try:
            cid = int(raw)
        except (TypeError, ValueError):
            continue
        if cid in session.rejected_ids or cid in session.inspect_ids_seen:
            continue
        if cid not in session.candidate_pool:
            continue
        ids.append(cid)
        if len(ids) >= STUDIO_DIRECTIONS_INSPECT_LIMIT:
            break

    if not ids:
        return {"status": "no_new_ids", "message": "No new valid catalog IDs to inspect."}

    rows = (
        await db.execute(select(DBImage).where(DBImage.id.in_(ids)))
    ).scalars().all()
    row_by_id = {int(r.id): r for r in rows}

    parts: list[Part] = [
        Part(text=(
            "Visual inspection batch — compare each Ducon catalog design below to the "
            "user's space photo from the start of this session. Decide which to keep, "
            "reject, or still need alternatives for."
        ))
    ]
    inspected: list[dict[str, Any]] = []

    async def _load_one(cid: int) -> tuple[Part | None, Part | None, dict[str, Any] | None]:
        row = row_by_id.get(cid)
        if not row:
            return None, None, None
        try:
            img = await load_ducon_image(row)
            img = normalize_user_image(_pil_bytes(img))
        except Exception as exc:
            session.rejected_ids.add(cid)
            print(f"[StudioDirections] inspect skip id={cid}: {exc}")
            return None, None, None

        meta = session.candidate_pool.get(cid, {})
        text_part = Part(text=(
            f"Catalog design catalog_id={cid} — {meta.get('name') or row.filename} "
            f"(theme: {meta.get('theme') or 'n/a'}, class: {meta.get('class') or 'n/a'})"
        ))
        info = {"catalog_id": cid, "name": meta.get("name") or row.filename}
        return text_part, _pil_part(img), info

    loaded = await asyncio.gather(*[_load_one(cid) for cid in ids])
    for text_part, image_part, info in loaded:
        if not text_part or not image_part or not info:
            continue
        cid = int(info["catalog_id"])
        parts.append(text_part)
        parts.append(image_part)
        session.inspect_ids_seen.add(cid)
        inspected.append(info)

    session.pending_visual_parts = parts
    return {
        "status": "images_attached",
        "inspected": inspected,
        "message": "Catalog images attached in the next turn for your visual review.",
    }


def _pil_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _run_shortlist(session: StudioDirectionsSession, directions: list[dict[str, Any]]) -> dict[str, Any]:
    already_ids = {int(d["catalog_id"]) for d in session.shortlisted}
    new_dirs: list[dict[str, Any]] = []
    for item in directions or []:
        if len(session.shortlisted) + len(new_dirs) >= _DIRECTION_COUNT:
            break
        try:
            cid = int(item.get("catalog_id"))
        except (TypeError, ValueError):
            continue
        if cid in already_ids or cid in session.rejected_ids:
            continue
        if cid not in session.candidate_pool:
            continue
        title = str(item.get("title") or "").strip()
        subtitle = str(item.get("subtitle") or "").strip()
        if not title:
            title = session.candidate_pool[cid].get("name") or f"Direction {len(session.shortlisted) + len(new_dirs) + 1}"
        new_dirs.append({
            "catalog_id": cid,
            "title": title,
            "subtitle": subtitle,
            "catalog": session.candidate_pool[cid],
        })
        already_ids.add(cid)

    start_slot = len(session.shortlisted)
    session.shortlisted.extend(new_dirs)
    remaining = _DIRECTION_COUNT - len(session.shortlisted)
    return {
        "new_count": len(new_dirs),
        "start_slot": start_slot,
        "total_shortlisted": len(session.shortlisted),
        "remaining_needed": remaining,
        "status": "need_more" if remaining > 0 else "complete",
    }


def _run_submit(session: StudioDirectionsSession, directions: list[dict[str, Any]]) -> dict[str, Any]:
    # Merge previously shortlisted + any new ones from this submit call.
    shortlisted_ids = {int(d["catalog_id"]) for d in session.shortlisted}
    remaining_needed = _DIRECTION_COUNT - len(session.shortlisted)
    new_dirs: list[dict[str, Any]] = []
    seen = set(shortlisted_ids)
    for item in directions or []:
        if len(new_dirs) >= remaining_needed:
            break
        try:
            cid = int(item.get("catalog_id"))
        except (TypeError, ValueError):
            continue
        if cid in seen or cid in session.rejected_ids:
            continue
        if cid not in session.candidate_pool:
            continue
        title = str(item.get("title") or "").strip()
        subtitle = str(item.get("subtitle") or "").strip()
        if not title:
            title = session.candidate_pool[cid].get("name") or f"Direction {len(session.shortlisted) + len(new_dirs) + 1}"
        new_dirs.append({
            "catalog_id": cid,
            "title": title,
            "subtitle": subtitle,
            "catalog": session.candidate_pool[cid],
        })
        seen.add(cid)

    merged = session.shortlisted + new_dirs
    if len(merged) < _DIRECTION_COUNT:
        return {
            "error": f"Need exactly {_DIRECTION_COUNT} valid directions; got {len(merged)}.",
            "accepted": len(merged),
        }

    session.submitted = merged[:_DIRECTION_COUNT]
    return {"status": "accepted", "count": len(session.submitted)}


def _format_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _selection_search_phrase(sel: dict) -> str:
    """Label plus distinctive tokens for embedding / text search queries."""
    label = (sel.get("label") or "").strip()
    tokens = [str(t or "").strip() for t in (sel.get("tokens") or []) if str(t or "").strip()]
    if not label and not tokens:
        return ""
    label_lower = label.lower()
    extra = [t for t in tokens if t.lower() not in label_lower][:6]
    if label and extra:
        return f"{label} ({', '.join(extra)})"
    if label:
        return label
    return ", ".join(extra)


def _format_selections_for_search(selections: list[dict]) -> list[str]:
    return [p for p in (_selection_search_phrase(s) for s in selections if isinstance(s, dict)) if p]


def _normalize_picker(data: dict) -> dict:
    """Support single or multi (OR) picker payloads from the studio wizard."""
    if not isinstance(data, dict):
        return {}
    selections = data.get("selections")
    if isinstance(selections, list) and selections:
        labels = [
            s.get("label") for s in selections
            if isinstance(s, dict) and s.get("label")
        ]
        tokens: list[str] = []
        for s in selections:
            if isinstance(s, dict):
                tokens.extend(s.get("tokens") or [])
        out = dict(data)
        out["label"] = " · ".join(labels) if labels else data.get("label", "")
        out["labels"] = labels
        out["tokens"] = list(dict.fromkeys(tokens))
        out["mode"] = data.get("mode") or "or"
        return out
    return data


def _subtitle_base(space: dict, style: dict) -> str:
    space_labels = space.get("labels") or ([space.get("label")] if space.get("label") else [])
    style_labels = style.get("labels") or ([style.get("label")] if style.get("label") else [])
    return " · ".join(x for x in [*space_labels, *style_labels] if x)


def _direction_image_url(direction: dict[str, Any]) -> str:
    """Resolve a public catalog image URL for SSE payloads (frontend fallback lookup)."""
    catalog = direction.get("catalog") or {}
    filename = catalog.get("filename") or ""
    meta = get_image_metadata(filename) if filename else None
    return str((meta or {}).get("url") or "")


def _direction_confirm_payload(direction: dict[str, Any], slot_index: int) -> dict[str, Any]:
    catalog = direction.get("catalog") or {}
    return {
        "catalog_id": direction["catalog_id"],
        "title": direction.get("title") or catalog.get("name") or f"Direction {slot_index + 1}",
        "subtitle": direction.get("subtitle") or "",
        "slot_index": slot_index,
        "filename": catalog.get("filename") or "",
        "image_url": _direction_image_url(direction),
    }


async def _emit_direction_confirms_stream(
    directions: list[dict[str, Any]],
    on_event: EventCallback,
    *,
    stagger_s: float = 0.12,
    start_slot: int = 0,
) -> None:
    """Stream direction confirms to the UI one slot at a time with a stagger."""
    await on_event({"type": "status", "phase": "curating", "filled_count": start_slot})
    for i, d in enumerate(directions):
        slot_idx = start_slot + i
        if slot_idx >= _DIRECTION_COUNT:
            break
        await on_event({
            "type": "direction_confirm",
            "directions": [_direction_confirm_payload(d, slot_idx)],
            "filled_count": slot_idx + 1,
        })
        if stagger_s > 0 and slot_idx < _DIRECTION_COUNT - 1:
            await asyncio.sleep(stagger_s)


def _fallback_directions(session: StudioDirectionsSession, space: dict, style: dict) -> list[dict[str, Any]]:
    """Deterministic fallback if the agent does not submit in time."""
    pool = list(session.candidate_pool.values())
    pool.sort(key=lambda r: (0 if (r.get("level") or "").lower() == "design" else 1, r.get("name") or ""))
    subtitle_base = _subtitle_base(space, style)
    titles = [
        "Primary match", "Alternative look", "Bold option", "Refined layout",
        "Open plan", "Statement entry", "Warm palette", "High contrast", "Resort feel",
    ]
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for rec in pool:
        cid = int(rec["catalog_id"])
        if cid in seen or cid in session.rejected_ids:
            continue
        seen.add(cid)
        out.append({
            "catalog_id": cid,
            "title": rec.get("name") or titles[len(out)],
            "subtitle": subtitle_base,
            "catalog": rec,
        })
        if len(out) >= _DIRECTION_COUNT:
            break
    return out


EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


def _build_presearch_queries(space: dict, style: dict) -> list[str]:
    """Build diverse pre-search queries from the user's space + style selections (OR)."""
    spaces = space.get("selections") or ([space] if space.get("label") else [])
    styles = style.get("selections") or ([style] if style.get("label") else [])
    if not spaces:
        spaces = [{"label": "outdoor space"}]
    if not styles:
        styles = [{"label": ""}]

    queries: list[str] = []
    for st in styles[:3]:
        for sp in spaces[:3]:
            s = _selection_search_phrase(sp) or "outdoor space"
            st_l = _selection_search_phrase(st)
            q = f"{st_l} {s}".strip()
            if q and q not in queries:
                queries.append(q)

    s0 = _selection_search_phrase(spaces[0]) or "outdoor space"
    st0 = _selection_search_phrase(styles[0]) if styles else ""
    extras = [
        f"Ducon luxury {s0} outdoor {st0} design".strip(),
        f"{st0} {s0} premium landscaping finished project outdoor living".strip(),
    ]
    for e in extras:
        if e and e not in queries:
            queries.append(e)
        if len(queries) >= 3:
            break
    return queries[:3]


async def _presearch_candidates(
    *,
    db: AsyncSession,
    collection,
    embedding_model: GeminiEmbeddingModel,
    session: StudioDirectionsSession,
    queries: list[str],
    limit_per_query: int,
    user_image_bytes: bytes,
    user_mime: str,
) -> None:
    """Run multiple searches in parallel to populate the candidate pool quickly.

    Embeddings are computed concurrently (each Gemini call takes ~1-2 s),
    then Chroma lookups run, and results are interleaved round-robin so the
    top hits from every query are prioritised over tail hits from any one query.
    """
    # Compute all embeddings concurrently — this is the main latency saving.
    embeddings = await asyncio.gather(*[
        asyncio.to_thread(
            embedding_model.get_multimodal_embedding,
            q, user_image_bytes, user_mime,
        )
        for q in queries
    ])

    # Chroma retrieval is CPU-bound but very fast; run in threads for safety.
    retrieval_results = await asyncio.gather(*[
        asyncio.to_thread(chromadb.retrieve, collection, emb, limit_per_query)
        for emb in embeddings
    ])

    # Interleave results round-robin: top-1 from each query, then top-2, etc.
    # This ensures diversity rather than flooding the pool from one query.
    filename_lists = [(r.get("ids") or [[]])[0] for r in retrieval_results]
    seen: set[str] = set()
    interleaved: list[str] = []
    max_len = max((len(fl) for fl in filename_lists), default=0)
    for i in range(max_len):
        for fl in filename_lists:
            if i < len(fl) and fl[i] not in seen:
                seen.add(fl[i])
                interleaved.append(fl[i])

    # Resolve DB rows in one query.
    row_by_filename = await _resolve_rows_by_filenames(db, interleaved)

    query_tag = " | ".join(queries)
    for filename in interleaved:
        row = row_by_filename.get(filename)
        if not row:
            continue
        cid = int(row.id)
        if cid not in session.candidate_pool and cid not in session.rejected_ids:
            session.candidate_pool[cid] = _serialize_catalog_row(row, query=query_tag)

    session.search_count += len(queries)
    print(
        f"[StudioDirections] pre-search: {len(queries)} parallel queries "
        f"→ {len(session.candidate_pool)} unique candidates"
    )


async def curate_studio_directions(
    *,
    db: AsyncSession,
    collection,
    embedding_model: GeminiEmbeddingModel,
    user_image_bytes: bytes,
    user_mime: str,
    space: dict[str, Any],
    style: dict[str, Any],
    on_event: EventCallback | None = None,
) -> dict[str, Any]:
    prompt_loader.ensure_prompts_loaded()
    client = get_gemini_client()
    model_id = studio_directions_model()
    thinking_level = studio_directions_thinking_level()
    thinking_label = thinking_level if thinking_level else "disabled"
    print(
        f"[StudioDirections] model={model_id!r} thinking={thinking_label!r}"
    )
    session = StudioDirectionsSession()
    space = _normalize_picker(space)
    style = _normalize_picker(style)

    # Normalize user image once — raw uploads can be 5–12 MB from phone cameras.
    # This reduces upload size for every Gemini embedding and generate call.
    try:
        _norm_pil = normalize_user_image(user_image_bytes)
        _norm_buf = io.BytesIO()
        _norm_pil.save(_norm_buf, format="JPEG", quality=88)
        user_image_bytes = _norm_buf.getvalue()
        user_mime = "image/jpeg"
        print(f"[StudioDirections] user image normalized to {len(user_image_bytes) // 1024} KB")
    except Exception as exc:
        print(f"[StudioDirections] image normalization skipped: {exc}")

    if on_event:
        await on_event({"type": "status", "phase": "reading"})

    # ── Phase 1: Parallel pre-search ──────────────────────────────────────────
    # Run multiple diverse queries concurrently before the agent loop starts.
    # This moves expensive Gemini embedding calls out of the agent's critical path.
    if on_event:
        await on_event({"type": "status", "phase": "searching", "search_count": _PRESEARCH_QUERIES})

    presearch_queries = _build_presearch_queries(space, style)
    await _presearch_candidates(
        db=db,
        collection=collection,
        embedding_model=embedding_model,
        session=session,
        queries=presearch_queries,
        limit_per_query=_PRESEARCH_LIMIT,
        user_image_bytes=user_image_bytes,
        user_mime=user_mime,
    )

    # ── Phase 2: Pre-inspect top candidates ───────────────────────────────────
    # Load catalog images for the best-ranked candidates in parallel so the
    # agent can shortlist on its first turn without calling inspect_designs.
    if on_event:
        await on_event({"type": "status", "phase": "inspecting"})

    top_ids = list(session.candidate_pool.keys())[:_PREINSPECT_TOP]
    if top_ids:
        await _run_inspect_designs(db=db, session=session, catalog_ids=top_ids)

    # ── Build conversation ─────────────────────────────────────────────────────
    brief = {
        "space_type": space,
        "style_direction": style,
        "space_search_phrases": _format_selections_for_search(
            [s for s in (space.get("selections") or []) if isinstance(s, dict)]
        ),
        "style_search_phrases": _format_selections_for_search(
            [s for s in (style.get("selections") or []) if isinstance(s, dict)]
        ),
        "goal": f"Curate exactly {_DIRECTION_COUNT} Ducon design directions for visualization.",
        "pre_search_done": True,
        "candidate_pool_size": len(session.candidate_pool),
        "instructions": (
            f"The candidate pool is already pre-populated with {len(session.candidate_pool)} results "
            f"from {len(presearch_queries)} parallel searches, and the top {len(top_ids)} catalog images "
            "are attached below for your immediate visual review. "
            "Start by calling shortlist_directions with the best matches you see. "
            "Only call ai_search or inspect_designs if you need more variety after shortlisting. "
            "Once all 9 are shortlisted, call submit_directions with all 9."
        ),
    }
    session.conversation.append(Content(role="user", parts=[
        Part.from_bytes(data=user_image_bytes, mime_type=user_mime),
        Part(text=f"User space photo.\n\nContext:\n{json.dumps(brief, ensure_ascii=False)}"),
    ]))

    # Attach pre-inspected catalog images so the agent sees them on turn 1.
    if session.pending_visual_parts:
        session.conversation.append(Content(role="user", parts=session.pending_visual_parts))
        session.pending_visual_parts = []

    config = _studio_directions_generate_config()

    agent_notes: list[str] = []

    for turn in range(1, int(cfg("STUDIO_DIRECTIONS_MAX_TURNS", STUDIO_DIRECTIONS_MAX_TURNS)) + 1):
        if session.submitted:
            break

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model_id,
            contents=session.conversation,
            config=config,
        )

        if not response.candidates:
            break

        model_content = response.candidates[0].content
        session.conversation.append(model_content)

        function_calls = [
            part.function_call
            for part in (model_content.parts or [])
            if part.function_call
        ]

        if not function_calls:
            text = response.text or ""
            if text.strip():
                agent_notes.append(text.strip()[:500])
            break

        response_parts: list[Part] = []
        for fc in function_calls:
            name = fc.name
            args = dict(fc.args or {})

            if name == "ai_search":
                if on_event:
                    await on_event({
                        "type": "status",
                        "phase": "searching",
                        "search_count": session.search_count + 1,
                    })
                result = await _run_ai_search(
                    db=db,
                    collection=collection,
                    embedding_model=embedding_model,
                    session=session,
                    query=str(args.get("query") or ""),
                    limit=int(args.get("limit") or STUDIO_DIRECTIONS_SEARCH_LIMIT),
                    user_image_bytes=user_image_bytes,
                    user_mime=user_mime,
                )
            elif name == "keyword_search":
                if on_event:
                    await on_event({"type": "status", "phase": "keyword_search"})
                opts = args.get("opts") if isinstance(args.get("opts"), dict) else {}
                result = await _run_keyword_search(
                    db=db,
                    session=session,
                    query=str(args.get("query") or ""),
                    opts=opts,
                    limit=int(args.get("limit") or 20),
                )
            elif name == "inspect_designs":
                if on_event:
                    await on_event({"type": "status", "phase": "inspecting"})
                result = await _run_inspect_designs(
                    db=db,
                    session=session,
                    catalog_ids=args.get("catalog_ids") or [],
                )
            elif name == "shortlist_directions":
                result = _run_shortlist(session, args.get("directions") or [])
                if on_event and result.get("new_count", 0) > 0:
                    start_slot = result["start_slot"]
                    new_dirs = session.shortlisted[start_slot: start_slot + result["new_count"]]
                    await _emit_direction_confirms_stream(new_dirs, on_event, start_slot=start_slot)
            elif name == "submit_directions":
                shortlisted_before = len(session.shortlisted)
                result = _run_submit(session, args.get("directions") or [])
                if (
                    on_event
                    and result.get("status") == "accepted"
                    and session.submitted
                    and not session.directions_streamed_to_ui
                ):
                    # Only emit slots that weren't already shown via shortlist_directions.
                    new_from_submit = session.submitted[shortlisted_before:]
                    if new_from_submit:
                        await _emit_direction_confirms_stream(
                            new_from_submit, on_event, start_slot=shortlisted_before
                        )
                    session.directions_streamed_to_ui = True
            else:
                result = {"error": f"Unknown tool: {name}"}

            response_parts.append(Part.from_function_response(name=name, response=result))

        session.conversation.append(Content(role="user", parts=response_parts))

        if session.pending_visual_parts:
            session.conversation.append(Content(role="user", parts=session.pending_visual_parts))
            session.pending_visual_parts = []

    if session.submitted:
        directions = session.submitted
    elif session.shortlisted:
        # Agent shortlisted some but never submitted — pad with fallback pool.
        shortlisted_ids = {int(d["catalog_id"]) for d in session.shortlisted}
        extra = [
            d for d in _fallback_directions(session, space, style)
            if int(d["catalog_id"]) not in shortlisted_ids
        ]
        remaining = _DIRECTION_COUNT - len(session.shortlisted)
        directions = session.shortlisted + extra[:remaining]
    else:
        directions = _fallback_directions(session, space, style)

    if len(directions) < _DIRECTION_COUNT:
        raise RuntimeError(
            f"Studio direction agent found only {len(directions)} candidates "
            f"(pool={len(session.candidate_pool)}, searches={session.search_count})."
        )

    if on_event and not session.directions_streamed_to_ui:
        shortlisted_count = len(session.shortlisted)
        new_dirs = directions[shortlisted_count:]
        if new_dirs:
            await _emit_direction_confirms_stream(new_dirs, on_event, start_slot=shortlisted_count)
        elif not shortlisted_count:
            await _emit_direction_confirms_stream(directions, on_event)
        session.directions_streamed_to_ui = True

    query_summary = "; ".join(
        sorted({d.get("catalog", {}).get("found_via_query", "") for d in directions if d.get("catalog")})
    ) or build_legacy_query(space, style)

    print(
        f"[StudioDirections] done — {len(directions)} directions, "
        f"{session.search_count} searches, pool={len(session.candidate_pool)}"
    )

    payload = {
        "query": query_summary,
        "directions": [
            _direction_confirm_payload(d, i)
            for i, d in enumerate(directions[:_DIRECTION_COUNT])
        ],
        "agent_notes": agent_notes,
        "search_count": session.search_count,
        "candidate_pool_size": len(session.candidate_pool),
        "model_used": model_id,
        "thinking_level": thinking_level,
    }
    if on_event:
        await on_event({"type": "done", **payload})
    return payload


async def stream_studio_directions_events(
    *,
    collection,
    embedding_model: GeminiEmbeddingModel,
    user_image_bytes: bytes,
    user_mime: str,
    space: dict[str, Any],
    style: dict[str, Any],
) -> AsyncIterator[str]:
    """SSE stream: candidates after each search, then done with final 9 directions."""
    from app.db.database import async_session_maker

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def on_event(payload: dict[str, Any]) -> None:
        await queue.put(payload)

    async def worker() -> None:
        try:
            async with async_session_maker() as db:
                await curate_studio_directions(
                    db=db,
                    collection=collection,
                    embedding_model=embedding_model,
                    user_image_bytes=user_image_bytes,
                    user_mime=user_mime,
                    space=space,
                    style=style,
                    on_event=on_event,
                )
        except Exception as exc:
            print(f"[StudioDirections] stream error: {exc}")
            await queue.put({"type": "error", "message": str(exc) or "Studio direction curation failed."})
        finally:
            await queue.put(None)

    KEEPALIVE_INTERVAL = 15.0  # seconds — resets Cloudflare's 100 s idle timeout

    task = asyncio.create_task(worker())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_INTERVAL)
            except asyncio.TimeoutError:
                # Send an SSE comment — ignored by clients, resets proxy idle timers
                yield ": keepalive\n\n"
                continue
            if item is None:
                break
            yield _format_sse(item)
            await asyncio.sleep(0)
    finally:
        await task


def build_legacy_query(space: dict, style: dict) -> str:
    spaces = space.get("selections") or ([space] if space.get("label") else [])
    styles = style.get("selections") or ([style] if style.get("label") else [])
    space_phrases = _format_selections_for_search([s for s in spaces if isinstance(s, dict)])
    style_phrases = _format_selections_for_search([s for s in styles if isinstance(s, dict)])
    space_part = ""
    if space_phrases:
        space_part = f" for {' or '.join(p.lower() for p in space_phrases)}"
    style_part = ""
    if style_phrases:
        style_part = f" in {' or '.join(p.lower() for p in style_phrases)} style"
    return (
        "Ducon premium outdoor living design inspiration"
        + space_part
        + style_part
        + ". Show finished landscape designs suitable for visualization on a user photo."
    )
