"""Catalog filter option summaries and Ducon orientation for agent prompts/tools."""

from __future__ import annotations

import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from app.image_utils import load_image_info

LEVEL_FILTER_HELP = (
    "All | Designs (full project designs, catalog level 5) | "
    "Products (modular standalone product tiles, levels 1-3) | "
    "Areas (area-level catalog entries, level 4)"
)

# Curated fallback when live metadata lacks class/type (or IMAGE_INFO is empty).
_FALLBACK_CLASSES = (
    "Walkways",
    "Sitting",
    "Artisan",
    "Furniture",
    "Planter box",
    "Area",
    "Kitchen",
    "Driveway",
    "Terrain",
    "Dining",
    "Playgrounds",
)
_FALLBACK_TYPES = (
    "Pergola",
    "Fountain",
    "Kitchen",
    "Planter box",
    "Driveway",
    "Driveway Parking",
    "Sitting",
    "Dining",
    "Seating",
    "Entrance",
    "Garden",
    "Walkways",
    "Counter Top",
    "Playground",
)
_FALLBACK_SPACES = (
    "pools",
    "terraces / patios / courtyards",
    "pergolas & shaded lounges",
    "pavers / hardscape",
    "outdoor kitchens & BBQ",
    "fountains / water features",
    "planters & landscaping",
    "walkways / driveways / entrances",
    "seating / majlis / dining",
)

_MAX_LIST = 28
_MAX_FEATURES = 20


def _clean_str(value: object) -> str:
    return str(value or "").strip()


def _sorted_unique(values: Iterable[str], *, limit: int = _MAX_LIST) -> tuple[str, ...]:
    cleaned = sorted({v for v in (_clean_str(x) for x in values) if v})
    return tuple(cleaned[:limit])


def _load_metadata_path_items() -> list[dict[str, Any]]:
    """Optional second source (METADATA_PATH / data/metadata.json) often has class/type/level."""
    candidates: list[Path] = []
    raw = (os.getenv("METADATA_PATH") or "").strip()
    if raw:
        candidates.append(Path(raw))
    candidates.append(Path("data") / "metadata.json")
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    return []


@lru_cache(maxsize=1)
def _catalog_items() -> tuple[dict[str, Any], ...]:
    items: list[dict[str, Any]] = []
    try:
        loaded = load_image_info()
        if isinstance(loaded, list):
            items.extend(i for i in loaded if isinstance(i, dict))
    except Exception:
        pass
    try:
        meta = _load_metadata_path_items()
        # Prefer richer taxonomy: merge by filename when both exist.
        by_fn: dict[str, dict[str, Any]] = {}
        for item in items:
            fn = _clean_str(item.get("filename"))
            if fn:
                by_fn[fn] = dict(item)
        for item in meta:
            fn = _clean_str(item.get("filename"))
            if not fn:
                continue
            base = by_fn.get(fn) or {}
            merged = dict(base)
            for key in ("class", "type", "theme", "level", "tags", "name", "description"):
                if item.get(key) not in (None, "", []):
                    merged[key] = item[key]
            if not merged.get("filename"):
                merged["filename"] = fn
            # Keep feature/element fields from IMAGE_INFO when present.
            for key in (
                "features_in_image",
                "elements_in_image",
                "feature_descriptions",
                "element_descriptions",
                "overall_description",
            ):
                if key in base and key not in merged:
                    merged[key] = base[key]
            by_fn[fn] = merged
        if by_fn:
            items = list(by_fn.values())
        elif meta and not items:
            items = meta
    except Exception:
        pass
    return tuple(items)


def clear_catalog_context_cache() -> None:
    """Test helper — drop cached taxonomy / orientation."""
    for fn in (
        _catalog_items,
        get_catalog_class_options,
        get_catalog_type_options,
        get_catalog_theme_options,
        get_catalog_feature_options,
        get_ducon_catalog_orientation,
    ):
        clearer = getattr(fn, "cache_clear", None)
        if callable(clearer):
            clearer()


@lru_cache(maxsize=1)
def get_catalog_class_options() -> tuple[str, ...]:
    try:
        return _sorted_unique(item.get("class") for item in _catalog_items() if item.get("class"))
    except Exception:
        return ()


@lru_cache(maxsize=1)
def get_catalog_type_options() -> tuple[str, ...]:
    try:
        return _sorted_unique(item.get("type") for item in _catalog_items() if item.get("type"))
    except Exception:
        return ()


@lru_cache(maxsize=1)
def get_catalog_theme_options() -> tuple[str, ...]:
    try:
        return _sorted_unique(
            (item.get("theme") for item in _catalog_items() if item.get("theme")),
            limit=16,
        )
    except Exception:
        return ()


@lru_cache(maxsize=1)
def get_catalog_feature_options() -> tuple[str, ...]:
    """Common scene/product motifs from features_in_image (IMAGE_INFO schema)."""
    try:
        counts: Counter[str] = Counter()
        for item in _catalog_items():
            feats = item.get("features_in_image") or []
            if isinstance(feats, dict):
                feats = list(feats.keys())
            if not isinstance(feats, list):
                continue
            for feat in feats:
                label = _clean_str(feat).lower()
                if label:
                    counts[label] += 1
        return tuple(label for label, _ in counts.most_common(_MAX_FEATURES))
    except Exception:
        return ()


def get_keyword_filter_context() -> str:
    classes = get_catalog_class_options() or _FALLBACK_CLASSES
    types = get_catalog_type_options() or _FALLBACK_TYPES
    class_line = ", ".join(classes)
    type_line = ", ".join(types)
    return (
        "Filter reference (theme/mood -> use ai_search; avoid guessing tags):\n"
        f"  • level: {LEVEL_FILTER_HELP}\n"
        f"  • class: {class_line}\n"
        f"  • category (matches catalog type): {type_line}"
    )


@lru_cache(maxsize=1)
def get_ducon_catalog_orientation() -> str:
    """
    Compact brand + taxonomy block for the designer agent system prompt.

    Prefer live catalog metadata (IMAGE_INFO + optional METADATA_PATH merge);
    fall back to a curated Ducon summary when taxonomy fields are sparse.
    """
    classes = get_catalog_class_options()
    types = get_catalog_type_options()
    themes = get_catalog_theme_options()
    features = get_catalog_feature_options()
    used_live = bool(classes or types or features)

    class_line = ", ".join(classes or _FALLBACK_CLASSES)
    type_line = ", ".join(types or _FALLBACK_TYPES)
    space_line = ", ".join(_FALLBACK_SPACES)
    theme_line = ", ".join(themes[:12]) if themes else ""
    feature_line = ", ".join(features) if features else ""

    source = "live catalog metadata" if used_live else "curated Ducon summary (live taxonomy sparse)"
    lines = [
        "## Ducon catalog orientation",
        f"(from {source})",
        "Ducon is a UAE premium outdoor living company. The library is a searchable "
        "catalog of completed project photos and modular product references.",
        f"Typical spaces / offerings: {space_line}.",
        f"Catalog tabs (keyword_search level): {LEVEL_FILTER_HELP}.",
        f"Product / area classes: {class_line}.",
        f"Product category / type values (keyword_search category): {type_line}.",
    ]
    if feature_line:
        lines.append(f"Common scene motifs in catalog photos: {feature_line}.")
    if theme_line:
        lines.append(f"Example project themes: {theme_line}.")
    lines.append(
        "Use these names in ai_search queries and keyword_search filters "
        "(level / class / category) — prefer dual search for concrete product types."
    )
    return "\n".join(lines)
