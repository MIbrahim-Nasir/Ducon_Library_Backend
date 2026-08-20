"""Catalog search helpers shared by the designer agent tools."""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import chromadb
from app.catalog_keyword_search import keyword_search_catalog
from app.db.models import Image as DBImage
from app.image_utils import get_image_metadata, load_ducon_image
from app.ml import GeminiEmbeddingModel


def filename_candidates(filenames: list[str]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for filename in filenames:
        for candidate in single_filename_candidates(filename):
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def single_filename_candidates(filename: str) -> list[str]:
    base, dot, ext = filename.rpartition(".")
    if not dot:
        return [filename]
    variants = [filename]
    for new_ext in ("jpg", "jpeg", "png", "webp"):
        variants.append(f"{base}.{new_ext}")
    result: list[str] = []
    seen: set[str] = set()
    for item in variants:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# Backward-compatible private aliases (benchmark / studio imports).
_filename_candidates = filename_candidates
_single_filename_candidates = single_filename_candidates


def fallback_queries(user_prompt: str) -> list[str]:
    base = user_prompt or "modern luxury outdoor living design"
    return [
        base,
        "premium outdoor terrace landscaping",
        "modern pool patio pergola Ducon design",
        "luxury villa outdoor seating pavers",
    ]


def fallback_generation_prompt(user_prompt: str) -> str:
    return (
        user_prompt
        or "Redesign the client's outdoor space as a premium Ducon concept with refined paving, "
        "balanced greenery, elegant lighting, and a realistic luxury villa exterior atmosphere."
    )


_fallback_queries = fallback_queries
_fallback_generation_prompt = fallback_generation_prompt


async def keyword_search(
    *,
    db: AsyncSession,
    query: str,
    level: Optional[str] = None,
    class_: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[list[str]] = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    result = await keyword_search_catalog(
        db,
        query=query,
        level=level,
        class_=class_,
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
                "label": hit.get("name") or hit.get("filename"),
                "metadata": get_image_metadata(hit.get("filename") or "") or {},
                "source": "keyword_search",
                "query": query,
            }
        )
    return hits


async def ai_search(
    *,
    db: AsyncSession,
    embedding_model: GeminiEmbeddingModel,
    collection,
    query: str,
    limit: int = 12,
) -> list[dict[str, Any]]:
    embedding = await asyncio.to_thread(embedding_model.get_text_embedding, query)
    result = await asyncio.to_thread(chromadb.retrieve, collection, embedding, limit)
    ids = (result.get("ids") or [[]])[0]
    if not ids:
        return []
    hits: list[dict[str, Any]] = []
    candidate_filenames = filename_candidates(ids)
    rows = (
        await db.execute(select(DBImage).where(DBImage.filename.in_(candidate_filenames)))
    ).scalars().all()
    row_by_filename: dict[str, DBImage] = {}
    for row in rows:
        for candidate in single_filename_candidates(row.filename):
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
                "label": row.name or row.filename,
                "metadata": {k: metadata[k] for k in list(metadata)[:8]} if isinstance(metadata, dict) else {},
                "source": "ai_search",
                "query": query,
            }
        )
    return hits


async def load_catalog_images_by_id(
    db: AsyncSession, ids: list[int]
) -> tuple[list[dict[str, Any]], list]:
    from PIL import Image

    infos: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    if not ids:
        return infos, images
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
