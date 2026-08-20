"""Designer agent system prompt composition."""
from __future__ import annotations

from app import prompt_loader
from app.catalog_filter_context import get_ducon_catalog_orientation
from app.designer import budgets


def build_system_prompt() -> str:
    """Compose the autonomous designer agent system instruction."""
    prompt_loader.ensure_prompts_loaded()
    agent = (prompt_loader.DESIGNER_AGENT_SYSTEM or "").strip()
    writer = (prompt_loader.DESIGNER_PROMPT_WRITER_SYSTEM or "").strip()
    orientation = (get_ducon_catalog_orientation() or "").strip()
    snap = budgets.budget_snapshot()
    budget_block = (
        "Soft harness budgets (safety ceilings — finish early when satisfied; do not pad):\n"
        f"- max model turns: {snap['max_turns']}\n"
        f"- max catalog searches (ai + keyword combined): {snap['max_searches']}\n"
        f"- max image generations: {snap['max_generations']}\n"
        f"- wall-clock seconds: {snap['wall_time_seconds'] or 'unlimited'}\n"
        f"- search hits per query (default / soft): {snap['search_limit']}\n"
        f"- max catalog references per generation (safety cap): {snap['max_references']}\n"
        f"- evaluate pass score threshold: {snap['pass_score']}\n"
    )
    parts = [p for p in (agent, orientation, budget_block, writer) if p]
    return "\n\n".join(parts)
