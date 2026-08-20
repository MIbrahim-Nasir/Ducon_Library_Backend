"""Long-running Ducon Designer Agent jobs.

Implementation lives in ``app.designer`` (autonomous tool-calling loop).
This module re-exports the public API so existing imports keep working.
"""
from __future__ import annotations

from app import storage  # noqa: F401 — patched by tests as app.designer_agent.storage
from app.db.database import async_session_maker  # noqa: F401 — patched by cross-worker tests
from app.designer import (  # noqa: F401
    DESIGNER_AGENT_DEFAULT_ASPECT_RATIO,
    DESIGNER_AGENT_DEFAULT_IMAGE_MODEL,
    DESIGNER_AGENT_MAX_GENERATIONS,
    DESIGNER_AGENT_MAX_REFERENCES,
    DESIGNER_AGENT_MAX_STEPS,
    DESIGNER_AGENT_MODEL,
    DESIGNER_AGENT_PASS_SCORE,
    DESIGNER_AGENT_SEARCH_LIMIT,
    JOBS,
    DesignerJob,
    _fallback_generation_prompt,
    _fallback_queries,
    _filename_candidates,
    _single_filename_candidates,
    cancel_job,
    create_job,
    drain_jobs_on_shutdown,
    emit,
    final_image_fields,
    get_job_for_user,
    run_designer_job,
    stream_job_events,
    stream_job_events_from_db,
)
from app.designer.jobs import (  # noqa: F401
    _SHUTDOWN_INTERRUPT_MSG,
    _TERMINAL_STATUSES,
    _USER_CANCEL_MSG,
    _check_cancelled,
    _enrich_generation_record,
    _persist_cancel_request,
    _persist_job_create,
    _persist_job_event,
    _persist_job_final,
)

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
    "async_session_maker",
    "storage",
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
    "_check_cancelled",
    "_enrich_generation_record",
    "_SHUTDOWN_INTERRUPT_MSG",
    "_USER_CANCEL_MSG",
    "_persist_cancel_request",
    "_persist_job_create",
    "_persist_job_event",
    "_persist_job_final",
]
