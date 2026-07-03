"""Catalog filter option summaries for agent tool descriptions."""

from __future__ import annotations

from functools import lru_cache

from app.image_utils import load_image_info

LEVEL_FILTER_HELP = (
    "All | Designs (full project designs, catalog level 5) | "
    "Products (modular standalone product tiles, levels 1–3) | "
    "Areas (area-level catalog entries, level 4)"
)


@lru_cache(maxsize=1)
def get_catalog_class_options() -> tuple[str, ...]:
    try:
        items = load_image_info()
        return tuple(sorted({
            str(item.get("class")).strip()
            for item in items
            if item.get("class")
        }))
    except Exception:
        return ()


def get_keyword_filter_context() -> str:
    classes = get_catalog_class_options()
    class_line = ", ".join(classes) if classes else "(see live catalog metadata)"
    return (
        "Filter reference (theme/mood → use AISearch; avoid guessing tags):\n"
        f"  • level: {LEVEL_FILTER_HELP}\n"
        f"  • class: {class_line}"
    )
