"""
Server-side keyword + advanced-filter search over Ducon catalog metadata.

Mirrors the frontend catalogFilter / librarySearch behaviour for agent tools.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Image as DBImage
from app.image_utils import image_info_by_filename

DESIGN_LEVELS = {5}
PRODUCT_LEVELS = {1, 2, 3}
AREA_LEVELS = {4}


def tokenize_search_query(raw: str) -> list[str]:
    if not raw or not isinstance(raw, str):
        return []
    return [t for t in raw.lower().strip().split() if t]


def _normalize_level(meta: dict[str, Any]) -> int | None:
    try:
        n = int(meta.get("level"))
        return n
    except (TypeError, ValueError):
        return None


def matches_level_filter(meta: dict[str, Any], level_filter: str | None) -> bool:
    if not level_filter or level_filter == "All":
        return True
    n = _normalize_level(meta)
    if n is None:
        return False
    if level_filter == "Designs":
        return n in DESIGN_LEVELS
    if level_filter == "Products":
        return n in PRODUCT_LEVELS
    if level_filter == "Areas":
        return n in AREA_LEVELS
    return True


def _word_boundary(haystack: str, token: str) -> bool:
    if not haystack or not token:
        return False
    pattern = rf"(^|[^a-z0-9]){re.escape(token)}([^a-z0-9]|$)"
    return re.search(pattern, haystack, re.I) is not None


def token_relevance(meta: dict[str, Any], row: Any, token: str) -> int:
    t = token.lower()
    best = 0

    name = str(getattr(row, "name", None) or meta.get("name") or meta.get("title") or "").lower()
    desc = str(meta.get("description") or "").lower()
    project = str(meta.get("project") or "").lower()
    filename = str(getattr(row, "filename", None) or "").lower()
    cls = str(meta.get("class") or "").lower()

    if name:
        if name == t:
            best = max(best, 1000)
        elif name.startswith(t):
            best = max(best, 930)
        elif _word_boundary(name, t):
            best = max(best, 880)
        elif t in name:
            best = max(best, 820)

    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    for tag in tags:
        g = str(tag).lower()
        if g == t:
            best = max(best, 790)
        elif g.startswith(t):
            best = max(best, 720)
        elif _word_boundary(g, t):
            best = max(best, 680)
        elif t in g:
            best = max(best, 620)

    if t in desc:
        best = max(best, 400)
    if t in filename:
        best = max(best, 450)
    if t in project:
        best = max(best, 380)
    if t in cls:
        best = max(best, 350)

    return best


def image_matches_tokens(meta: dict[str, Any], row: Any, tokens: list[str]) -> bool:
    if not tokens:
        return True
    return all(token_relevance(meta, row, tok) > 0 for tok in tokens)


def _meta_passes_filters(
    meta: dict[str, Any],
    *,
    level: str | None,
    class_: str | None,
    category: str | None,
    tag_list: list[str],
    tag_logic_norm: str,
) -> bool:
    if class_ and class_ != "All" and str(meta.get("class") or "") != class_:
        return False
    if category and category != "All":
        img_type = str(meta.get("type") or meta.get("class") or "")
        if img_type != category:
            return False
    if level and level != "All" and not matches_level_filter(meta, level):
        return False
    if tag_list:
        img_tags = [str(t) for t in (meta.get("tags") or [])]
        if tag_logic_norm == "AND":
            if not all(t in img_tags for t in tag_list):
                return False
        elif not any(t in img_tags for t in tag_list):
            return False
    return True


def _filenames_matching_filters(
    meta_by_filename: dict[str, dict],
    *,
    level: str | None,
    class_: str | None,
    category: str | None,
    tag_list: list[str],
    tag_logic_norm: str,
) -> set[str] | None:
    """Return allowed filenames when advanced filters are set; else None (= all)."""
    has_filter = (
        (level and level != "All")
        or (class_ and class_ != "All")
        or (category and category != "All")
        or bool(tag_list)
    )
    if not has_filter:
        return None
    allowed: set[str] = set()
    for filename, meta in meta_by_filename.items():
        if _meta_passes_filters(
            meta or {},
            level=level,
            class_=class_,
            category=category,
            tag_list=tag_list,
            tag_logic_norm=tag_logic_norm,
        ):
            allowed.add(filename)
    return allowed


def _score_rows(
    rows: list[tuple[int, str | None, str, str]],
    *,
    tokens: list[str],
    meta_by_filename: dict[str, dict],
    level: str | None,
    class_: str | None,
    category: str | None,
    tag_list: list[str],
    tag_logic_norm: str,
) -> list[tuple[tuple[int, str | None, str, str], dict[str, Any], int]]:
    """CPU-bound filter/score — safe to run in a worker thread."""
    hits: list[tuple[tuple[int, str | None, str, str], dict[str, Any], int]] = []
    for row_id, name, filename, url in rows:
        meta = meta_by_filename.get(filename) or {}
        row = type("Row", (), {"id": row_id, "name": name, "filename": filename, "url": url})()
        if not _meta_passes_filters(
            meta,
            level=level,
            class_=class_,
            category=category,
            tag_list=tag_list,
            tag_logic_norm=tag_logic_norm,
        ):
            continue
        if not image_matches_tokens(meta, row, tokens):
            continue
        score = sum(token_relevance(meta, row, t) for t in tokens) if tokens else 0
        hits.append(((row_id, name, filename, url), meta, score))
    return hits


async def keyword_search_catalog(
    db: AsyncSession,
    *,
    query: str = "",
    level: str | None = None,
    class_: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    tag_logic: str = "OR",
    cross_tab: bool = False,
    limit: int = 24,
) -> dict[str, Any]:
    """Filter catalog rows by keyword tokens and advanced metadata filters."""
    tokens = tokenize_search_query(query)
    tag_list = [str(t) for t in (tags or []) if str(t).strip()]
    tag_logic_norm = "AND" if str(tag_logic or "OR").upper() == "AND" else "OR"
    limit = max(1, min(int(limit or 24), 48))

    meta_by_filename = image_info_by_filename()
    allowed = _filenames_matching_filters(
        meta_by_filename,
        level=level,
        class_=class_,
        category=category,
        tag_list=tag_list,
        tag_logic_norm=tag_logic_norm,
    )

    stmt = select(DBImage)
    if allowed is not None:
        if not allowed:
            return {
                "query": query.strip() or "Catalog filter",
                "hits": [],
                "hit_count": 0,
                "tag_logic": tag_logic_norm,
                "cross_tab": bool(cross_tab),
                "note": (
                    "Keyword/filter results from catalog metadata. Prefer ai_search for semantic "
                    "style discovery unless the user asked for exact names or product filters."
                ),
            }
        stmt = stmt.where(DBImage.filename.in_(sorted(allowed)))

    db_rows = (await db.execute(stmt)).scalars().all()
    rows = [(int(r.id), r.name, r.filename, r.url) for r in db_rows]

    # Token scoring is CPU-bound; keep the event loop free for SSE.
    hits = await asyncio.to_thread(
        _score_rows,
        rows,
        tokens=tokens,
        meta_by_filename=meta_by_filename,
        level=level,
        class_=class_,
        category=category,
        tag_list=tag_list,
        tag_logic_norm=tag_logic_norm,
    )

    if tokens:
        hits.sort(
            key=lambda item: (
                -item[2],
                str(item[0][1] or item[0][2]).lower(),
            ),
        )
    else:
        hits.sort(key=lambda item: str(item[0][1] or item[0][2]).lower())

    sliced = hits[:limit]
    records = [
        {
            "id": row_id,
            "name": name or filename,
            "filename": filename,
            "url": url,
            "class": meta.get("class"),
            "theme": meta.get("theme"),
            "project": meta.get("project"),
            "level": meta.get("level"),
            "tags": meta.get("tags") or [],
            "_type": "catalog_image",
        }
        for (row_id, name, filename, url), meta, _ in sliced
    ]

    label_parts = [p for p in [query.strip(), level, class_, category, *(tag_list or []),
                               (f"Match {tag_logic_norm}" if tag_logic_norm != "OR" else None)] if p]
    return {
        "query": " · ".join(label_parts) or "Catalog filter",
        "hits": records,
        "hit_count": len(records),
        "tag_logic": tag_logic_norm,
        "cross_tab": bool(cross_tab),
        "note": (
            "Keyword/filter results from catalog metadata. Prefer ai_search for semantic "
            "style discovery unless the user asked for exact names or product filters."
        ),
    }
