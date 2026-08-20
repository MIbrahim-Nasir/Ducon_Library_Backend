"""Production Ducon designer — autonomous tool-calling agent package."""
from __future__ import annotations

from app.designer.budgets import (
    DEFAULT_ASPECT_RATIO as DESIGNER_AGENT_DEFAULT_ASPECT_RATIO,
    DEFAULT_IMAGE_MODEL as DESIGNER_AGENT_DEFAULT_IMAGE_MODEL,
    DEFAULT_MAX_GENERATIONS as DESIGNER_AGENT_MAX_GENERATIONS,
    DEFAULT_MAX_TURNS as DESIGNER_AGENT_MAX_STEPS,
    DEFAULT_MODEL as DESIGNER_AGENT_MODEL,
    DEFAULT_PASS_SCORE as DESIGNER_AGENT_PASS_SCORE,
    DEFAULT_SEARCH_LIMIT as DESIGNER_AGENT_SEARCH_LIMIT,
    max_references as _max_refs,
)
from app.designer.catalog import (
    _fallback_generation_prompt,
    _fallback_queries,
    _filename_candidates,
    _single_filename_candidates,
)
from app.designer.jobs import (
    JOBS,
    DesignerJob,
    cancel_job,
    create_job,
    drain_jobs_on_shutdown,
    emit,
    final_image_fields,
    get_job_for_user,
    stream_job_events,
    stream_job_events_from_db,
)
from app.designer.runner import run_designer_job

# Env-style aliases kept for external readers / older code.
DESIGNER_AGENT_MAX_REFERENCES = _max_refs()

__all__ = [
    "JOBS",
    "DesignerJob",
    "DESIGNER_AGENT_MODEL",
    "DESIGNER_AGENT_MAX_STEPS",
    "DESIGNER_AGENT_MAX_GENERATIONS",
    "DESIGNER_AGENT_SEARCH_LIMIT",
    "DESIGNER_AGENT_MAX_REFERENCES",
    "DESIGNER_AGENT_PASS_SCORE",
    "DESIGNER_AGENT_DEFAULT_IMAGE_MODEL",
    "DESIGNER_AGENT_DEFAULT_ASPECT_RATIO",
    "cancel_job",
    "create_job",
    "drain_jobs_on_shutdown",
    "emit",
    "final_image_fields",
    "get_job_for_user",
    "run_designer_job",
    "stream_job_events",
    "stream_job_events_from_db",
    "_fallback_generation_prompt",
    "_fallback_queries",
    "_filename_candidates",
    "_single_filename_candidates",
]
