"""Autonomous designer agent tool-calling loop."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.designer import budgets
from app.designer.jobs import DesignerJob, check_cancelled, emit
from app.designer.llm import append_tool_message, chat_with_tools
from app.designer.prompts import build_system_prompt
from app.designer.tools import DesignerToolbox
from app.ml import GeminiEmbeddingModel


def _status_shows_max_turns() -> bool:
    """Advertise turn ceilings in UI status only when debugging."""
    level = (os.getenv("LOG_LEVEL") or "").strip().lower()
    if level in {"debug", "10"}:
        return True
    try:
        from app.admin.settings_store import cfg_bool

        if cfg_bool("LIVE_DEBUG", False):
            return True
    except Exception:
        pass
    return logging.getLogger().isEnabledFor(logging.DEBUG)


def turn_status_message(turn: int, turn_cap: int) -> str:
    """User-facing progress line — clean by default; max only in debug."""
    if _status_shows_max_turns():
        return f"Designer agent working (turn {turn}, max {turn_cap})."
    return f"Designer agent working (turn {turn})."


async def run_agent_loop(
    *,
    job: DesignerJob,
    db: AsyncSession,
    user_image: Image.Image,
    user_prompt: str,
    embedding_model: GeminiEmbeddingModel,
    collection,
    image_model: str,
    aspect_ratio: Optional[str],
) -> tuple[DesignerToolbox, str, int]:
    """Run the tool-calling loop until finish / budget / cancel.

    Returns ``(toolbox, stop_reason, turns_used)``.
    """
    from app.observability.langfuse_client import (
        _truncate,
        flush as langfuse_flush,
        observe_span,
        start_trace,
        update_trace_output,
    )

    system = build_system_prompt()
    toolbox = DesignerToolbox(
        job=job,
        db=db,
        user_image=user_image,
        embedding_model=embedding_model,
        collection=collection,
        image_model=image_model,
        aspect_ratio=aspect_ratio,
    )
    tools = toolbox.declarations()
    turn_cap = budgets.max_turns()
    wall_cap_s = budgets.wall_time_seconds()
    started = time.monotonic()

    brief = (user_prompt or "").strip() or (
        "(no brief supplied — design a premium Ducon upgrade for this space)"
    )
    wall_label = "unlimited" if wall_cap_s <= 0 else f"{wall_cap_s}s"
    task_text = (
        f"Client brief: {brief}\n"
        f"Preferred aspect ratio: {toolbox.aspect_ratio}\n"
        f"Soft budgets (stop early when done — do not pad to the max): "
        f"up to {turn_cap} model turns, {toolbox.max_searches} searches, "
        f"{toolbox.max_generations} generations, {wall_label} wall clock; "
        f"up to {toolbox.max_references} catalog refs per generate_image; "
        f"~{toolbox.search_limit} hits per search by default.\n"
        'The attached image is the client\'s space photo (reference as "user_photo").\n'
        "Work freely within those budgets: plan and refine as needed, search with both "
        "ai_search and keyword_search when useful (especially for concrete product types), "
        "inspect and choose the best refs (not a tiny fixed shortlist), generate, "
        "evaluate/revise, and call finish when you have a client-ready design."
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "text": task_text, "images": [user_image]}
    ]

    stop_reason = "max_turns_reached"
    turns_used = 0
    nudged = False
    consecutive_model_errors = 0
    turn = 0

    job_tags = ["designer", f"job_id:{job.id}"]
    with start_trace(
        "designer-job",
        as_type="agent",
        user_id=job.user_id,
        session_id=str(job.id),
        tags=job_tags,
        metadata={
            "agent": "designer",
            "job_id": job.id,
            "image_model": image_model,
            "aspect_ratio": aspect_ratio,
        },
        input={"brief": brief[:1500], "job_id": job.id},
    ):
        try:
            while True:
                turn += 1
                turns_used = turn
                check_cancelled(job)
                if turn > turn_cap:
                    stop_reason = "max_turns_reached"
                    break
                if wall_cap_s > 0 and (time.monotonic() - started) > wall_cap_s:
                    stop_reason = "wall_clock_budget"
                    break

                await emit(
                    job,
                    "status",
                    message=turn_status_message(turn, turn_cap),
                    turn=turn,
                    max_turns=turn_cap,
                )

                try:
                    resp = await chat_with_tools(
                        system=system,
                        messages=messages,
                        tools=tools,
                        user_id=job.user_id,
                    )
                except Exception as exc:
                    consecutive_model_errors += 1
                    await emit(job, "tool_error", tool="model", error=str(exc), turn=turn)
                    if consecutive_model_errors >= 2:
                        stop_reason = f"model_error: {exc}"
                        break
                    await asyncio.sleep(2)
                    continue
                consecutive_model_errors = 0

                text = str(resp.get("text") or "").strip()
                tool_calls = resp.get("tool_calls") or []
                if text:
                    await emit(job, "assistant_message", text=text, turn=turn)
                messages.append(
                    {
                        "role": "assistant",
                        "text": text or None,
                        "tool_calls": tool_calls,
                        "provider_state": resp.get("provider_state"),
                    }
                )

                if not tool_calls:
                    if not toolbox.generations and not nudged:
                        nudged = True
                        messages.append(
                            {
                                "role": "user",
                                "text": (
                                    "You have not produced a design yet. Continue: submit_plan if needed, "
                                    "search the catalog, inspect references, call generate_image. "
                                    "If you truly cannot proceed, call finish and explain why."
                                ),
                            }
                        )
                        continue
                    stop_reason = "agent_stopped_calling_tools"
                    break

                finished = False
                for tc in tool_calls:
                    check_cancelled(job)
                    tool_name = str(tc.get("name") or "")
                    await emit(
                        job,
                        "tool_call",
                        turn=turn,
                        id=tc.get("id"),
                        name=tool_name,
                        args=tc.get("args") or {},
                    )
                    with observe_span(
                        f"designer-tool:{tool_name or 'unknown'}",
                        as_type="tool",
                        input={"name": tool_name, "args": tc.get("args") or {}},
                        metadata={"job_id": job.id, "turn": turn},
                        tags=job_tags,
                    ) as tool_span:
                        result, images, image_caption = await toolbox.execute(
                            tool_name,
                            tc.get("args") or {},
                        )
                        try:
                            safe_result = result if isinstance(result, dict) else {"result": result}
                            tool_span.update(
                                output={
                                    "result": _truncate(safe_result, 1200),
                                    "image_count": len(images),
                                }
                            )
                        except Exception:
                            pass
                    await emit(
                        job,
                        "tool_result",
                        turn=turn,
                        id=tc.get("id"),
                        name=tc.get("name"),
                        result=result,
                        output_images=(result.get("output_images") if isinstance(result, dict) else None),
                        image_count=len(images),
                    )
                    append_tool_message(
                        messages,
                        tool_call_id=tc.get("id"),
                        name=tool_name,
                        result=result if isinstance(result, dict) else {"result": result},
                        images=images,
                        image_caption=image_caption,
                    )
                    if tc.get("name") == "finish":
                        finished = True
                # Long jobs: flush periodically so reload/cancel still ships spans.
                if turn % 3 == 0:
                    langfuse_flush()
                if finished:
                    stop_reason = "agent_called_finish"
                    break
        finally:
            try:
                update_trace_output(
                    {
                        "stop_reason": stop_reason,
                        "turns_used": turns_used,
                        "searches_used": toolbox.searches_used,
                        "generations": len(toolbox.generations),
                        "finished": bool(toolbox.finish_args),
                    }
                )
            except Exception:
                pass
            langfuse_flush()

    return toolbox, stop_reason, turns_used
