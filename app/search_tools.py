"""
Shared catalog search tool schemas for chat, voice, studio directions, and designer agents.

AISearch  — semantic / visual discovery (preferred default).
KeywordSearch / keyword_search — exact names, modular product tiles, advanced filters only.
"""

from __future__ import annotations

from typing import Any

from app.catalog_filter_context import get_keyword_filter_context


AI_SEARCH_WHEN = (
    "PREFERRED for almost all catalog discovery: inspiration, mood, style, layout ideas, "
    "vague descriptions, themes, materials, project vibes, and 'show me designs like…'. "
    "Uses semantic + visual matching. "
    "Accepts text query, an optional user image (upload/space photo), or both. "
    "Text-only, image-only, and text+image each produce different vector-search rankings — "
    "use text for named styles/materials, image when matching a space photo's look, "
    "and both when the user wants designs that fit their photo AND a described vibe."
)

AI_SEARCH_IMAGE_PARAM = {
    "type": "string",
    "description": (
        "Optional user space / reference image for visual similarity search. "
        "Pass a chat upload id, UploadImage upload_id, or other frontend-resolvable "
        "image ref the user provided. Omit for text-only search. "
        "When set without query, runs image-only embedding search. "
        "When set with query, runs multimodal (text+image) embedding search — "
        "results differ from text-only or image-only."
    ),
}

KEYWORD_SEARCH_WHEN = (
    "Use ONLY when the user needs exact catalog filtering — NOT for inspiration or vague "
    "style discovery (use AISearch / ai_search instead). Appropriate when they want: "
    "(1) individual modular product images (Products level), "
    "(2) a specific design or product by exact name or filename fragment, or "
    "(3) advanced filter combinations (level + class + category + tags + match mode). "
    "Do NOT use KeywordSearch for theme/mood browsing — that is AISearch."
)

KEYWORD_OPTS_PROPERTIES: dict[str, Any] = {
    "level": {
        "type": "string",
        "enum": ["All", "Designs", "Products", "Areas"],
        "description": (
            "Catalog tab / image type. 'Products' = modular standalone product tiles "
            "(pavers, pergolas, fountains). 'Designs' = full project designs. "
            "'Areas' = area-level entries. 'All' = search every tab (cross-tab)."
        ),
    },
    "class": {
        "type": "string",
        "description": (
            "Exact Ducon class metadata value. See filter reference in tool description "
            "for allowed class names. Omit or use All to skip."
        ),
    },
    "category": {
        "type": "string",
        "description": (
            "Product or area type name within the Products/Areas tabs "
            "(e.g. 'Fountain', 'Pergola', 'Paver') — matches catalog `type` metadata."
        ),
    },
    "tags": {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            "Optional tag metadata filter. Avoid unless the user named a specific tag — "
            "the catalog has many tags and guessing rarely helps."
        ),
    },
    "tagLogic": {
        "type": "string",
        "enum": ["OR", "AND", "or", "and"],
        "description": (
            "Tag match mode (advanced filter toggle): OR = any tag matches (default), "
            "AND = all tags must match."
        ),
    },
    "matchMode": {
        "type": "string",
        "enum": ["OR", "AND", "or", "and"],
        "description": "Alias for tagLogic — same as the UI 'Match: OR / AND' control.",
    },
    "crossTab": {
        "type": "boolean",
        "description": (
            "When true, search across Designs, Products, and Areas simultaneously "
            "(same as active keyword/filter search in the library). Default true when "
            "level is omitted or 'All'."
        ),
    },
}

KEYWORD_OPTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Advanced catalog filters applied together with query text.",
    "properties": KEYWORD_OPTS_PROPERTIES,
}


def keyword_search_parameters(*, require_query: bool = False) -> dict[str, Any]:
    props: dict[str, Any] = {
        "query": {
            "type": "string",
            "description": (
                "Keyword text matched against image name, filename, project, tags, and "
                "description. Use for exact or partial names (e.g. 'Outdoor Kitchen with "
                "Water Feature', 'marble coping', 'pergola'). Leave empty when only filters "
                "in opts are needed."
            ),
        },
        "opts": KEYWORD_OPTS_SCHEMA,
    }
    required: list[str] = ["query"] if require_query else []
    return {"type": "object", "properties": props, "required": required}


# ── Interactions API (chat agent) ─────────────────────────────────────────────

def ai_search_interactions_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "AISearch",
        "description": (
            "Semantic visual search of the Ducon catalog. "
            f"{AI_SEARCH_WHEN} "
            "Returns catalog image records shown inline in chat as a labeled AI slider. "
            "For concrete product types (pergola, fountain, paver…), pair with KeywordSearch "
            "in the same turn. For vague/themed browse: call once, then reply briefly. "
            "Provide at least one of query or image_ref."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language or descriptive search query. "
                        "Optional when image_ref is provided (image-only search)."
                    ),
                },
                "image_ref": AI_SEARCH_IMAGE_PARAM,
                "show_user": {
                    "type": "boolean",
                    "description": (
                        "True when the user should see the AI search results UI. "
                        "False when searching only for internal reference gathering."
                    ),
                },
                "presentation": {
                    "type": "string",
                    "enum": ["show_user", "internal"],
                    "description": "Legacy alias for show_user. Prefer show_user boolean.",
                },
            },
            "required": [],
        },
    }


def keyword_search_interactions_tool(*, require_query: bool = False, for_chat: bool = False) -> dict[str, Any]:
    chat_dual = ""
    if for_chat:
        chat_dual = (
            " CHAT DUAL SEARCH: when the user asks for a concrete product or item type "
            "(pergola, fountain, paver, coping, kitchen, fire pit, gazebo, etc.), call "
            "AISearch AND KeywordSearch in the same turn — AISearch for matching designs, "
            "KeywordSearch with the same term and opts.level='Products' for modular product "
            "tiles. Use AISearch alone for vague, themed, or complex multi-criteria requests."
        )
    filter_ctx = get_keyword_filter_context()
    return {
        "type": "function",
        "name": "KeywordSearch",
        "description": (
            "Exact keyword + metadata filter search on the Ducon catalog. "
            f"{KEYWORD_SEARCH_WHEN} "
            f"{filter_ctx} "
            "In chat, results appear inline as a separate labeled slider."
            f"{chat_dual}"
        ),
        "parameters": keyword_search_parameters(require_query=require_query),
    }


# ── Gemini Live / function_declarations (voice) ───────────────────────────────

def ai_search_live_declaration() -> dict[str, Any]:
    return {
        "name": "AISearch",
        "description": (
            "Semantic visual search of the Ducon catalog. "
            f"{AI_SEARCH_WHEN} "
            "Opens the AI search modal and returns CatalogImage records "
            "{id, name, filename, class, theme, project, tags, url, _type:'catalog_image'}. "
            "Treat results as metadata — call get_image to inspect pixels when needed. "
            "Provide at least one of query or image_ref (upload_id from UploadImage or chat)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language or descriptive search query. "
                        "Optional when image_ref is provided (image-only search)."
                    ),
                },
                "image_ref": AI_SEARCH_IMAGE_PARAM,
                "show_user": {
                    "type": "boolean",
                    "description": (
                        "True when the user should see search results in the UI. "
                        "False for internal reference gathering only."
                    ),
                },
            },
            "required": [],
        },
    }


def keyword_search_live_declaration(*, require_query: bool = False) -> dict[str, Any]:
    filter_ctx = get_keyword_filter_context()
    return {
        "name": "KeywordSearch",
        "description": (
            "Exact keyword + metadata filter search on the catalog grid. "
            f"{KEYWORD_SEARCH_WHEN} "
            f"{filter_ctx} "
            "Applies filters in the library UI. Prefer AISearch for discovery."
        ),
        "parameters": keyword_search_parameters(require_query=require_query),
    }


# ── Studio directions curator (snake_case backend tool) ───────────────────────

def keyword_search_studio_spec() -> dict[str, Any]:
    filter_ctx = get_keyword_filter_context()
    return {
        "name": "keyword_search",
        "description": (
            "Exact catalog keyword/filter lookup on metadata — NOT semantic discovery. "
            f"{KEYWORD_SEARCH_WHEN} "
            f"{filter_ctx} "
            "Prefer ai_search for style-matching direction curation."
        ),
        "parameters": keyword_search_parameters(require_query=False),
    }
