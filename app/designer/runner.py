"""Designer job runner — entry point for autonomous agent jobs."""
from __future__ import annotations

import asyncio
import io
from typing import Any, Optional

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app import storage
from app.designer import budgets
from app.designer.jobs import (
    JOBS,
    DesignerJob,
    _SHUTDOWN_INTERRUPT_MSG,
    _TERMINAL_STATUSES,
    _USER_CANCEL_MSG,
    _dbg,
    _persist_job_create,
    _persist_job_final,
    emit,
    enrich_generation_record,
)
from app.designer.loop import run_agent_loop
from app.designer.tools import parse_selected_generation
from app.image_utils import normalize_user_image
from app.ml import GeminiEmbeddingModel


def _pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return buf.getvalue()


async def run_designer_job(
    *,
    job: DesignerJob,
    db: AsyncSession,
    embedding_model: GeminiEmbeddingModel,
    collection,
    user_image_bytes: bytes,
    user_prompt: Optional[str],
    model: str = budgets.DEFAULT_IMAGE_MODEL,
    aspect_ratio: Optional[str] = None,
) -> None:
    """Autonomous tool-calling designer agent entry (replaces fixed pipeline)."""
    await _persist_job_create(job)
    try:
        aspect = aspect_ratio if aspect_ratio is not None else budgets.default_aspect_ratio()
        image_model = model or budgets.image_model()
        _dbg(
            "[DESIGNER ▶ JOB START]",
            {
                "job_id": job.id,
                "user_id": job.user_id,
                "model": image_model,
                "aspect_ratio": aspect,
                "prompt": (user_prompt or "")[:500],
                "input_image_bytes": len(user_image_bytes),
                "budgets": budgets.budget_snapshot(),
                "loop": "agentic_tool_calling",
            },
        )
        job.status = "running"
        await emit(job, "status", message="Analyzing the client's space image.")

        user_image = await asyncio.to_thread(normalize_user_image, user_image_bytes)
        user_prompt = (user_prompt or "").strip()

        # Persist client space photo for slider "before" (no watermark).
        user_image_url = None
        try:
            input_image_bytes = await asyncio.to_thread(_pil_to_bytes, user_image, "PNG")
            input_stored_key = await storage.asave_designer_input(
                job.user_id, job.id, input_image_bytes
            )
            user_image_url = storage.get_designer_input_url(job.id, input_stored_key)
            await emit(
                job,
                "input_image",
                user_image={"url": user_image_url, "label": "Your space"},
                message="Client space photo received.",
            )
        except Exception as exc:
            _dbg(f"[DESIGNER ⚠ input image persist failed: {exc}]")

        toolbox, stop_reason, turns_used = await run_agent_loop(
            job=job,
            db=db,
            user_image=user_image,
            user_prompt=user_prompt,
            embedding_model=embedding_model,
            collection=collection,
            image_model=image_model,
            aspect_ratio=aspect,
        )

        if stop_reason == "cancelled" or job.cancel_requested:
            raise asyncio.CancelledError()

        selected = parse_selected_generation(
            (toolbox.finish_args or {}).get("selected_generation"),
            toolbox.generations,
        )
        if selected is None:
            raise RuntimeError(
                f"Designer job finished without a generated candidate (stop={stop_reason})."
            )

        # Prefer last generation for UI "best_generation" (matches prior contract);
        # also expose best-scored when evaluations exist.
        last = toolbox.generations[-1]
        best = selected
        best_score = -1.0
        for g in toolbox.generations:
            ev = g.get("evaluation") or {}
            try:
                score = float(ev.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            if score > best_score:
                best_score = score
                best = g

        summary = ""
        if toolbox.finish_args:
            summary = str(toolbox.finish_args.get("summary") or "")
        if not summary:
            summary = (
                (toolbox.plan or {}).get("design_direction")
                or "Your Ducon design preview is ready."
            )

        attempts: list[dict[str, Any]] = []
        for g in toolbox.generations:
            attempts.append(
                {
                    "attempt": g.get("index"),
                    "generation": g.get("generation"),
                    "evaluation": g.get("evaluation"),
                    "references": g.get("references") or [],
                    "prompt": g.get("prompt"),
                }
            )

        last_generation = enrich_generation_record(dict(last.get("generation") or {}))
        best_generation = enrich_generation_record(dict(best.get("generation") or {}))

        job.status = "completed"
        job.final = {
            "job_id": job.id,
            "user_image": (
                {"url": user_image_url, "label": "Your space"}
                if user_image_url
                else None
            ),
            "best_generation": last_generation,
            "best_scored_generation": best_generation,
            "best_evaluation": best.get("evaluation"),
            "references": best.get("references") or selected.get("references") or [],
            "attempts": attempts,
            "summary": summary,
            "agent": {
                "loop": "agentic",
                "stop_reason": stop_reason,
                "turns_used": turns_used,
                "searches_used": toolbox.searches_used,
                "plan": toolbox.plan,
            },
        }
        await emit(job, "final", **job.final)
        await _persist_job_final(job, final=job.final)

    except asyncio.CancelledError:
        if job.status not in _TERMINAL_STATUSES:
            if job.cancel_requested:
                msg = _USER_CANCEL_MSG
            else:
                msg = job.error or _SHUTDOWN_INTERRUPT_MSG
            job.status = "cancelled"
            job.error = msg
            try:
                await emit(job, "cancelled", message=msg)
                await _persist_job_final(job, error=msg)
            except Exception:
                _dbg("[DESIGNER ⚠ cancel persist failed]")
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        _dbg("[DESIGNER ✖ ERROR]", repr(exc))
        await emit(job, "error", message=str(exc))
        await _persist_job_final(job, error=str(exc))
    finally:
        await job.queue.put(None)
        if job.status in {"completed", "failed", "cancelled"}:
            JOBS.pop(job.id, None)
        try:
            from app.observability.langfuse_client import flush as langfuse_flush
            langfuse_flush()
        except Exception:
            pass
