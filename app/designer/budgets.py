"""Designer agent budget knobs (env / settings catalog)."""
from __future__ import annotations

import os

from app.admin.settings_store import cfg

# Defaults — live values via cfg() / env.
DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_MAX_TURNS = 12
DEFAULT_MAX_GENERATIONS = 3
DEFAULT_MAX_SEARCHES = 12
DEFAULT_SEARCH_LIMIT = 12  # richer shortlist per query (model chooses)
DEFAULT_MAX_REFERENCES = 7  # catalog refs per generate_image (safety cap)
DEFAULT_PASS_SCORE = 7.5
DEFAULT_IMAGE_MODEL = "flash"
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_WALL_TIME_S = 900  # 15 minutes
DEFAULT_MAX_IMAGES_PER_GET = 4
SEARCH_LIMIT_HARD_MAX = 15  # absolute ceiling on hits returned per search


def designer_model() -> str:
    return str(cfg("DESIGNER_AGENT_MODEL", DEFAULT_MODEL))


def max_turns() -> int:
    """Max model turns in the tool-calling loop (was DESIGNER_AGENT_MAX_STEPS)."""
    return max(1, int(cfg("DESIGNER_AGENT_MAX_STEPS", DEFAULT_MAX_TURNS)))


def max_generations() -> int:
    return max(1, min(int(cfg("DESIGNER_AGENT_MAX_GENERATIONS", DEFAULT_MAX_GENERATIONS)), 12))


def max_searches() -> int:
    return max(1, int(cfg("DESIGNER_AGENT_MAX_SEARCHES", DEFAULT_MAX_SEARCHES)))


def search_limit() -> int:
    raw = cfg("DESIGNER_AGENT_SEARCH_LIMIT", os.getenv("DESIGNER_AGENT_SEARCH_LIMIT", str(DEFAULT_SEARCH_LIMIT)))
    return max(1, min(int(raw), SEARCH_LIMIT_HARD_MAX))


def max_references() -> int:
    raw = cfg(
        "DESIGNER_AGENT_MAX_REFERENCES",
        os.getenv("DESIGNER_AGENT_MAX_REFERENCES", str(DEFAULT_MAX_REFERENCES)),
    )
    return max(1, min(int(raw), 12))


def pass_score() -> float:
    raw = cfg(
        "DESIGNER_AGENT_PASS_SCORE",
        os.getenv("DESIGNER_AGENT_PASS_SCORE", str(DEFAULT_PASS_SCORE)),
    )
    return float(raw)


def image_model() -> str:
    return str(cfg("DESIGNER_AGENT_IMAGE_MODEL", DEFAULT_IMAGE_MODEL))


def default_aspect_ratio() -> str:
    return str(
        cfg(
            "DESIGNER_AGENT_ASPECT_RATIO",
            os.getenv("DESIGNER_AGENT_ASPECT_RATIO", DEFAULT_ASPECT_RATIO),
        )
    )


def wall_time_seconds() -> int:
    """Hard wall-clock budget for a designer job (0 = unlimited)."""
    raw = cfg("DESIGNER_AGENT_WALL_TIME_SECONDS", DEFAULT_WALL_TIME_S)
    value = int(raw)
    if value <= 0:
        return 0
    return max(60, value)


def budget_snapshot() -> dict:
    return {
        "max_turns": max_turns(),
        "max_generations": max_generations(),
        "max_searches": max_searches(),
        "search_limit": search_limit(),
        "max_references": max_references(),
        "pass_score": pass_score(),
        "wall_time_seconds": wall_time_seconds(),
    }
