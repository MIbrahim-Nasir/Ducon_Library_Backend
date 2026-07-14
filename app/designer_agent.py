"""
Long-running Ducon Designer Agent jobs.

This module implements a small durable-enough agent loop without a DB migration:
jobs live in memory, progress is streamed over SSE, and generated images are
persisted through the existing `generations` table/storage flow.

Agent loop:
1. Analyze the client's uploaded space image.
2. Create a design plan and search strategy.
3. Search the Ducon catalog for multiple reference directions.
4. Generate one or more candidates using generate_multi_image.
5. Visually evaluate each candidate and retry with revised prompts if needed.
6. Return the best generation and a concise design summary.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from google.genai.types import GenerateContentConfig
from PIL import Image
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app import chromadb, storage
from app.db.database import async_session_maker
from app.db.models import DesignerJobRow, Generation, Image as DBImage, User
from app.gemini import (
    get_gemini_client,
    verify_prompt as _verify_prompt,
    PROMPT_VERIFY_MAX_ROUNDS,
)
from app.image_utils import get_image_metadata, normalize_user_image
from app.image_utils import load_ducon_image
from app.catalog_keyword_search import keyword_search_catalog
from app.ml import GeminiEmbeddingModel
from app.tool_generate_image import ImageDescriptor, generate_multi_image
from app.prompt_generator_session import DesignerPromptSession
from app import prompt_loader
from app import llm_provider


from app.admin.settings_store import cfg

DESIGNER_AGENT_MODEL = "gemini-3.5-flash"           # default; live via cfg("DESIGNER_AGENT_MODEL", ...)
DESIGNER_AGENT_MAX_STEPS = 12
DESIGNER_AGENT_MAX_GENERATIONS = 3
DESIGNER_AGENT_SEARCH_LIMIT = int(os.getenv("DESIGNER_AGENT_SEARCH_LIMIT", "5"))
DESIGNER_AGENT_MAX_REFERENCES = int(os.getenv("DESIGNER_AGENT_MAX_REFERENCES", "6"))
DESIGNER_AGENT_PASS_SCORE = float(os.getenv("DESIGNER_AGENT_PASS_SCORE", "7.5"))
DESIGNER_AGENT_DEFAULT_IMAGE_MODEL = "flash"
DESIGNER_AGENT_DEFAULT_ASPECT_RATIO = os.getenv("DESIGNER_AGENT_ASPECT_RATIO", "16:9")
_LIVE_DEBUG: bool = False


def _designer_model() -> str:
    return cfg("DESIGNER_AGENT_MODEL", DESIGNER_AGENT_MODEL)


def _designer_max_steps() -> int:
    return int(cfg("DESIGNER_AGENT_MAX_STEPS", DESIGNER_AGENT_MAX_STEPS))


def _designer_max_generations() -> int:
    return int(cfg("DESIGNER_AGENT_MAX_GENERATIONS", DESIGNER_AGENT_MAX_GENERATIONS))


def _designer_image_model() -> str:
    return cfg("DESIGNER_AGENT_IMAGE_MODEL", DESIGNER_AGENT_DEFAULT_IMAGE_MODEL)

_CRITICAL_SECTION_KEYS = (
    "A1_pov",
    "A2_structures",
    "A3_scene",
    "B1_area_products",
    "B2_fixed_products",
    "B3_zones",
    "C1_no_extra",
    "C2_no_missing",
    "F1_mark_followthrough",
    "F2_mark_cleanup",
)


def _enrich_generation_record(gen: dict[str, Any]) -> dict[str, Any]:
    """Return generation dict with a fresh presigned URL for the UI."""
    enriched = dict(gen)
    gen_id = enriched.get("id")
    if gen_id is not None:
        enriched["signed_url"] = storage.get_generation_url(int(gen_id), enriched.get("url"))
    return enriched


def _dbg(*args) -> None:
    if cfg("LIVE_DEBUG", _LIVE_DEBUG):
        print(*args)


def _image_summary(img: Image.Image) -> dict[str, Any]:
    return {"mode": img.mode, "size": img.size}


def _pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return buf.getvalue()


def _usage(response) -> dict | None:
    usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
    if usage is None:
        return None
    return usage.model_dump(exclude_none=True) if hasattr(usage, "model_dump") else usage


def _section_value(value: object) -> str:
    return str(value or "").strip().lower()


def _sections_pass(section_results: object) -> bool:
    if not isinstance(section_results, dict):
        return False
    for key in _CRITICAL_SECTION_KEYS:
        value = _section_value(section_results.get(key))
        # "na" is acceptable only when the section genuinely does not apply
        # (e.g. no fixed product requested for B2).
        if value not in {"pass", "na", "n/a", "not_applicable", "not applicable"}:
            return False
    return True


def _passes_quality_gate(data: dict[str, Any]) -> bool:
    try:
        score = float(data.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    if score < DESIGNER_AGENT_PASS_SCORE:
        return False
    if not bool(data.get("passed")):
        return False
    return _sections_pass(data.get("section_results"))


@dataclass
class DesignerJob:
    id: str
    user_id: int
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    events: list[dict[str, Any]] = field(default_factory=list)
    queue: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)
    task: Optional[asyncio.Task] = None
    final: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    cancel_requested: bool = False


JOBS: dict[str, DesignerJob] = {}


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _persist_job_create(job: DesignerJob) -> None:
    """Insert the job row so other workers can see it immediately."""
    try:
        async with async_session_maker() as db:
            db.add(DesignerJobRow(
                id=job.id,
                user_id=job.user_id,
                status=job.status,
            ))
            await db.commit()
    except Exception:
        _dbg("[DESIGNER ⚠ persist create failed]")


async def _persist_job_event(job: DesignerJob, payload: dict[str, Any]) -> None:
    """Append one event to the job's JSONB event log + bump status/updated_at.

    Uses a raw SQL array concat so it is a single, atomic round-trip (no
    read-modify-write race between concurrent emits). Best-effort: a DB failure
    here never breaks the in-memory live stream on the owning worker.
    """
    try:
        async with async_session_maker() as db:
            await db.execute(
                text(
                    "UPDATE designer_jobs "
                    "SET events = events || CAST(:evt AS jsonb), "
                    "    status = :status, "
                    "    updated_at = NOW() "
                    "WHERE id = :id"
                ),
                {"evt": json.dumps([payload], ensure_ascii=False), "status": job.status, "id": job.id},
            )
            await db.commit()
    except Exception:
        _dbg("[DESIGNER ⚠ persist event failed]")


async def _persist_job_final(
    job: DesignerJob,
    *,
    final: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Persist the terminal state (completed/failed/cancelled) + final/error."""
    try:
        async with async_session_maker() as db:
            row = await db.get(DesignerJobRow, job.id)
            if row is None:
                # Upsert terminal row if create never landed (avoids stuck pollers).
                row = DesignerJobRow(
                    id=job.id,
                    user_id=job.user_id,
                    status=job.status,
                    events=list(job.events or []),
                )
                db.add(row)
            row.status = job.status
            row.updated_at = datetime.now(timezone.utc)
            if final is not None:
                row.final = final
            if error is not None:
                row.error = error
            await db.commit()
    except Exception:
        _dbg("[DESIGNER ⚠ persist final failed]")


async def emit(job: DesignerJob, event_type: str, **data: Any) -> None:
    payload = {
        "type": event_type,
        "job_id": job.id,
        "status": job.status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    _dbg("[DESIGNER EVENT]", payload)
    job.events.append(payload)
    await job.queue.put(_sse(payload))
    # Mirror to Postgres so workers that don't own this job can still serve
    # GET /designer/jobs/{id} and a polling /events stream. Awaited inline to
    # preserve event ordering in the DB log (jobs emit ~30-50 events total, so
    # the few-ms round-trip per emit is negligible vs the 30-90s job runtime).
    await _persist_job_event(job, payload)


def create_job(user: User) -> DesignerJob:
    job = DesignerJob(id=uuid.uuid4().hex, user_id=int(user.id))
    JOBS[job.id] = job
    return job


async def get_job_for_user(job_id: str, user_id: int) -> Optional[DesignerJob]:
    """Resolve a job for a user, preferring the live in-memory registry (this
    worker owns the running job) and falling back to the persisted row.

    Returns a DesignerJob populated from the DB row when the job is not in this
    worker's JOBS registry (cross-worker case). The returned object's
    ``events`` list reflects everything persisted so far.
    """
    job = JOBS.get(job_id)
    if job is not None and job.user_id == user_id:
        return job
    try:
        async with async_session_maker() as db:
            row = await db.get(DesignerJobRow, job_id)
            if row is None or row.user_id != user_id:
                return None
            restored = DesignerJob(id=row.id, user_id=int(row.user_id))
            restored.status = row.status
            restored.created_at = row.created_at.isoformat() if row.created_at else restored.created_at
            restored.events = list(row.events or [])
            restored.final = row.final
            restored.error = row.error
            restored.cancel_requested = bool(row.cancel_requested)
            # Not in this worker's registry; no live queue/task.
            restored.task = None
            return restored
    except Exception:
        _dbg("[DESIGNER ⚠ get_job_for_user db read failed]")
        return None


_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


async def stream_job_events(job: DesignerJob):
    """Live SSE stream for a job owned by THIS worker (has a live queue)."""
    yield ": connected\n\n"
    for event in job.events:
        yield _sse(event)
    while True:
        try:
            item = await asyncio.wait_for(job.queue.get(), timeout=10)
        except asyncio.TimeoutError:
            yield ": keep-alive\n\n"
            continue
        if item is None:
            break
        yield item


async def stream_job_events_from_db(job_id: str, user_id: int):
    """Polling SSE stream for a job owned by ANOTHER worker.

    Replays all persisted events, then polls the JSONB event log for new ones
    every couple of seconds until the job reaches a terminal status. Events
    arrive with up to ~2s latency versus the owning worker's live stream, which
    is acceptable for a regression-safe cross-worker experience.
    """
    yield ": connected\n\n"
    sent = 0
    idle = 0
    while True:
        try:
            async with async_session_maker() as db:
                row = await db.get(DesignerJobRow, job_id)
        except Exception:
            yield ": keep-alive\n\n"
            await asyncio.sleep(2)
            continue
        if row is None or row.user_id != user_id:
            # Vanished mid-stream — stop.
            break
        events = list(row.events or [])
        if len(events) > sent:
            for event in events[sent:]:
                yield _sse(event)
            sent = len(events)
            idle = 0
        else:
            idle += 1
        if row.status in _TERMINAL_STATUSES:
            break
        await asyncio.sleep(2)


async def cancel_job(job_id: str, user_id: int) -> bool:
    """Request cancellation of a designer job.

    Returns True iff the job exists and belongs to ``user_id``. Sets the
    in-memory cancel flag (and cancels the running task) when this worker owns
    the job, and always persists ``cancel_requested = TRUE`` so a cancel
    request landing on a different worker is durably recorded.

    Cross-worker cancel won't interrupt the owning worker's in-flight task
    (that process isn't reachable from here), but the flag is stored for
    inspection and the job will still terminate on its own.
    """
    job = JOBS.get(job_id)
    if job is not None and job.user_id == user_id:
        job.cancel_requested = True
        if job.task and not job.task.done():
            job.task.cancel()
        await _persist_cancel_request(job_id, user_id)
        return True

    # Cross-worker: confirm ownership via the DB row, then persist the flag.
    try:
        async with async_session_maker() as db:
            row = await db.get(DesignerJobRow, job_id)
            if row is None or row.user_id != user_id:
                return False
        await _persist_cancel_request(job_id, user_id)
        return True
    except Exception:
        _dbg("[DESIGNER ⚠ cross-worker cancel db read failed]")
        return False


async def _persist_cancel_request(job_id: str, user_id: int) -> None:
    try:
        async with async_session_maker() as db:
            await db.execute(
                text(
                    "UPDATE designer_jobs "
                    "SET cancel_requested = TRUE, updated_at = NOW() "
                    "WHERE id = :id AND user_id = :uid"
                ),
                {"id": job_id, "uid": user_id},
            )
            await db.commit()
    except Exception:
        _dbg("[DESIGNER ⚠ persist cancel failed]")


async def run_designer_job(
    *,
    job: DesignerJob,
    db: AsyncSession,
    embedding_model: GeminiEmbeddingModel,
    collection,
    user_image_bytes: bytes,
    user_prompt: Optional[str],
    model: str = DESIGNER_AGENT_DEFAULT_IMAGE_MODEL,
    aspect_ratio: Optional[str] = DESIGNER_AGENT_DEFAULT_ASPECT_RATIO,
) -> None:
    """Main long-running designer loop."""
    await _persist_job_create(job)
    try:
        _dbg(
            "[DESIGNER ▶ JOB START]",
            {
                "job_id": job.id,
                "user_id": job.user_id,
                "model": model,
                "aspect_ratio": aspect_ratio,
                "prompt": (user_prompt or "")[:500],
                "input_image_bytes": len(user_image_bytes),
                "budgets": {
                    "max_steps": DESIGNER_AGENT_MAX_STEPS,
                    "max_generations": DESIGNER_AGENT_MAX_GENERATIONS,
                    "search_limit": DESIGNER_AGENT_SEARCH_LIMIT,
                    "pass_score": DESIGNER_AGENT_PASS_SCORE,
                },
                "context": (
                    "Prompt writer uses DesignerPromptSession (multi-turn Gemini history). "
                    "Verifier and QC remain stateless one-shot calls."
                ),
            },
        )
        job.status = "running"
        await emit(job, "status", message="Analyzing the client's space image.")

        user_image = await asyncio.to_thread(normalize_user_image, user_image_bytes)
        _dbg("[DESIGNER ▶ INPUT IMAGE]", _image_summary(user_image))
        user_prompt = (user_prompt or "").strip()

        # Persist the client's original space photo so the result viewer can show
        # it as the slider "before" (and cross-worker / page-refresh clients can
        # fetch it via GET /designer/jobs/{job_id}/input-image). Stored WITHOUT a
        # watermark — it is the client's own upload, not a Ducon generation.
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
            # Persistence is best-effort: a failure here must not abort the run.
            # The generation still proceeds; only the slider "before" is affected.
            _dbg(f"[DESIGNER ⚠ input image persist failed: {exc}]")
            user_image_url = None

        plan = await _analyze_and_plan(user_image, user_prompt, user_id=job.user_id)
        await emit(job, "plan", plan=plan)

        search_queries = plan.get("search_queries") or []
        if not search_queries:
            search_queries = _fallback_queries(user_prompt)

        keyword_specs = plan.get("keyword_search_queries") or []
        keyword_refs, keyword_pruned = await _search_references_by_keyword(
            db=db,
            specs=keyword_specs[:3],
            job=job,
        )

        references, pruned_references = await _search_references(
            db=db,
            collection=collection,
            embedding_model=embedding_model,
            queries=search_queries[:4],
            job=job,
        )

        if keyword_refs:
            seen_ids = {int(r["id"]) for r in references}
            merged = list(keyword_refs)
            for ref in references:
                if int(ref["id"]) not in seen_ids:
                    merged.append(ref)
            references = merged[:DESIGNER_AGENT_MAX_REFERENCES]
            pruned_references = keyword_pruned + pruned_references
        if not references:
            raise RuntimeError("No Ducon catalog references were found for this design run.")
        await emit(
            job,
            "reference_board",
            selected=references,
            removed=pruned_references[:20],
            removed_count=len(pruned_references),
            message=f"Selected {len(references)} catalog references; {len(pruned_references)} alternate or duplicate hits were skipped during search.",
        )
        _dbg("[DESIGNER ◀ REFERENCES]", references)

        attempts: list[dict[str, Any]] = []
        best: Optional[dict[str, Any]] = None
        base_prompt = plan.get("generation_prompt") or _fallback_generation_prompt(user_prompt)
        prompt_session = DesignerPromptSession()
        last_attempt_evaluation: Optional[dict[str, Any]] = None

        max_generations = max(1, min(DESIGNER_AGENT_MAX_GENERATIONS, 5))
        for attempt_no in range(1, max_generations + 1):
            _check_cancelled(job)
            await emit(
                job,
                "generation_started",
                attempt=attempt_no,
                max_attempts=max_generations,
                message=f"Generating design candidate {attempt_no}.",
            )

            chosen_refs = references[: min(3, len(references))]
            reference_images = await _load_reference_images(db, chosen_refs)
            verify_labels = [
                "client space photo",
                *[ref["label"] for ref in chosen_refs],
            ]
            session_images = [user_image]
            session_labels = ["client space photo"]
            if reference_images:
                session_images.extend(reference_images)
                session_labels.extend([ref["label"] for ref in chosen_refs])

            if attempt_no == 1:
                prompt_plan = {
                    **plan,
                    "selected_generation_references": [
                        {
                            "id": ref.get("id"),
                            "name": ref.get("name"),
                            "label": ref.get("label"),
                            "metadata": ref.get("metadata"),
                        }
                        for ref in chosen_refs
                    ],
                }
                prompt = await prompt_session.generate_initial(
                    images=session_images,
                    labels=session_labels,
                    plan=prompt_plan,
                    design_hint=base_prompt,
                    attempt_no=attempt_no,
                )
            else:
                eval_source = last_attempt_evaluation or (best or {}).get("evaluation") or {}
                prompt = await prompt_session.revise_from_qc(
                    eval_source,
                    context_label=f"Designer job attempt {attempt_no} — outer QC retry",
                )

            # ── Pre-generation prompt verification (stateless) ────────────────
            if reference_images:
                verify_images = session_images
                for vround in range(PROMPT_VERIFY_MAX_ROUNDS):
                    _check_cancelled(job)
                    try:
                        v_passed, v_issues, v_improved = await asyncio.to_thread(
                            _verify_prompt, verify_images, verify_labels, prompt
                        )
                        if v_passed:
                            _dbg(f"[DESIGNER ▶ PROMPT_VERIFY] Passed on round {vround + 1}")
                            break
                        if v_improved:
                            _dbg(
                                f"[DESIGNER ▶ PROMPT_VERIFY] Round {vround + 1}: "
                                f"{len(v_issues)} issue(s), improving prompt"
                            )
                            prompt = v_improved
                            prompt_session.record_external_revision(
                                prompt,
                                source=f"pre-generation verifier round {vround + 1}",
                                issues=v_issues,
                            )
                        else:
                            _dbg(
                                f"[DESIGNER ▶ PROMPT_VERIFY] Round {vround + 1}: "
                                "failed, no improvement available"
                            )
                            break
                    except Exception as ve:
                        _dbg(f"[DESIGNER ▶ PROMPT_VERIFY] Skipped round {vround + 1}: {ve}")
                        break
                await emit(job, "status", message=f"Prompt verified. Starting generation attempt {attempt_no}.")

            if prompt_session.last_prompt != prompt:
                prompt_session.record_external_revision(
                    prompt,
                    source="pre-generation verifier final prompt",
                )

            descriptors = [
                ImageDescriptor(label="client space photo", type="file", pil_image=user_image)
            ]
            for ref in chosen_refs:
                descriptors.append(
                    ImageDescriptor(
                        label=ref["label"],
                        type="catalog_id",
                        source=str(ref["id"]),
                    )
                )

            result = await generate_multi_image(
                user_id=job.user_id,
                prompt=prompt,
                descriptors=descriptors,
                model=model,
                aspect_ratio=aspect_ratio,
                db=db,
                output_prefix=f"designer_job_{job.id[:8]}_{attempt_no}",
                prompt_session=prompt_session,
                # Designer jobs run their own explicit verifier + QC loop so
                # feedback is visible and goes through the job prompt session.
                enable_verify=False,
            )
            _dbg("[DESIGNER ◀ GENERATION RESULT]", result)
            await emit(job, "generation_done", attempt=attempt_no, generation=result)

            prompt = prompt_session.last_prompt

            generated_image = await _load_generation_image(db, int(result["id"]))
            # reference_images already loaded above for the pre-gen verifier
            if not reference_images:
                reference_images = await _load_reference_images(db, chosen_refs)
            evaluation = await _evaluate_generation(
                user_image=user_image,
                generated_image=generated_image,
                plan=plan,
                references=chosen_refs,
                reference_images=reference_images,
                prompt=prompt,
                user_id=job.user_id,
            )
            _dbg("[DESIGNER ◀ EVALUATION]", evaluation)
            attempt = {
                "attempt": attempt_no,
                "generation": result,
                "evaluation": evaluation,
                "references": chosen_refs,
                "prompt": prompt,
            }
            attempts.append(attempt)
            await emit(job, "evaluation", attempt=attempt_no, evaluation=evaluation)
            last_attempt_evaluation = evaluation

            new_score = float(evaluation.get("score", 0))
            best_score = float(best["evaluation"].get("score", 0)) if best else -1.0
            # Prefer higher score; on a tie prefer the later attempt (more refined).
            if best is None or new_score > best_score or (
                new_score == best_score and attempt_no > best.get("attempt", 0)
            ):
                best = attempt

            if evaluation.get("passed"):
                await emit(job, "status", message="The design candidate passed quality review.")
                break

            if attempt_no < max_generations:
                await emit(
                    job,
                    "retry",
                    attempt=attempt_no + 1,
                    reason=evaluation.get("improvements") or "Improving realism and design fit.",
                )

        if best is None:
            raise RuntimeError("Designer job finished without a generated candidate.")

        final_summary = await _summarize_final(plan=plan, best=best, attempts=attempts, user_id=job.user_id)
        _dbg("[DESIGNER ◀ FINAL SUMMARY]", final_summary)
        # UI shows the last generated candidate (final attempt), not highest-scored.
        last_generation = _enrich_generation_record(dict(attempts[-1]["generation"]))
        job.status = "completed"
        job.final = {
            "job_id": job.id,
            # Client's original space photo → slider "before". May be None if
            # persistence failed mid-run; the frontend falls back to references.
            "user_image": (
                {"url": user_image_url, "label": "Your space"}
                if user_image_url
                else None
            ),
            "best_generation": last_generation,
            "best_scored_generation": _enrich_generation_record(dict(best["generation"])),
            "best_evaluation": best["evaluation"],
            "references": best["references"],
            "attempts": attempts,
            "summary": final_summary,
        }
        await emit(job, "final", **job.final)
        await _persist_job_final(job, final=job.final)

    except asyncio.CancelledError:
        job.status = "cancelled"
        await emit(job, "cancelled", message="Designer job was cancelled.")
        await _persist_job_final(job)
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        _dbg("[DESIGNER ✖ ERROR]", repr(exc))
        await emit(job, "error", message=str(exc))
        await _persist_job_final(job, error=str(exc))
    finally:
        await job.queue.put(None)
        # Drop from the owning worker's in-memory registry once terminal; cross-worker
        # clients and reconnects read from Postgres via get_job_for_user().
        if job.status in {"completed", "failed", "cancelled"}:
            JOBS.pop(job.id, None)


async def _analyze_and_plan(user_image: Image.Image, user_prompt: str, *, user_id: Optional[int] = None) -> dict[str, Any]:
    client = get_gemini_client()
    prompt_loader.ensure_prompts_loaded()
    prompt = (
        f"{prompt_loader.DESIGNER_ANALYZE_PLAN}\n\n"
        f'User request/suggestions: {user_prompt or "No specific instructions. Design independently."}'
    )
    _dbg(
        "[DESIGNER ▶ GEMINI analyze_plan]",
        {
            "model": DESIGNER_AGENT_MODEL,
            "image": _image_summary(user_image),
            "prompt_chars": len(prompt),
            "prompt_preview": prompt[:1000],
            "response_mime_type": "application/json",
        },
    )
    if llm_provider.use_claude():
        text = await llm_provider.agenerate_text(
            "",
            [llm_provider.pil_image_block(user_image), llm_provider.text_block(prompt)],
        ) or "{}"
    else:
        response = await client.aio.models.generate_content(
            model=DESIGNER_AGENT_MODEL,
            contents=[user_image, prompt],
            config=GenerateContentConfig(response_mime_type="application/json"),
        )
        text = response.text or "{}"
        from app.admin.usage_helpers import record_from_response
        record_from_response(
            response,
            agent="designer",
            model=DESIGNER_AGENT_MODEL,
            user_id=user_id,
        )
        _dbg(
            "[DESIGNER ◀ GEMINI analyze_plan]",
            {"usage": _usage(response), "text_chars": len(text), "text_preview": text[:1200]},
        )
    try:
        return llm_provider.parse_json_text(text) if llm_provider.use_claude() else json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {
            "space_analysis": text[:1000],
            "design_direction": user_prompt or "Create a refined Ducon outdoor concept.",
            "preserve": [],
            "opportunities": [],
            "search_queries": _fallback_queries(user_prompt),
            "keyword_search_queries": [],
            "generation_prompt": _fallback_generation_prompt(user_prompt),
            "success_criteria": ["photorealistic", "fits the existing space", "uses Ducon references clearly"],
        }


async def _search_references_by_keyword(
    *,
    db: AsyncSession,
    specs: list[Any],
    job: DesignerJob,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exact name / filter lookups — modular products and named catalog items."""
    refs: list[dict[str, Any]] = []
    pruned: list[dict[str, Any]] = []
    seen: set[int] = set()

    for raw in specs or []:
        _check_cancelled(job)
        if isinstance(raw, str):
            spec = {"query": raw}
        elif isinstance(raw, dict):
            spec = raw
        else:
            continue

        query = str(spec.get("query") or spec.get("name") or "").strip()
        opts = spec.get("opts") if isinstance(spec.get("opts"), dict) else spec
        level = opts.get("level") if isinstance(opts, dict) else None
        class_ = opts.get("class") if isinstance(opts, dict) else None
        category = opts.get("category") if isinstance(opts, dict) else None
        tags = opts.get("tags") if isinstance(opts, dict) else None
        tag_logic = (
            opts.get("tagLogic") or opts.get("matchMode") or opts.get("tag_logic") or "OR"
        ) if isinstance(opts, dict) else "OR"
        cross_tab = bool(
            (opts.get("crossTab") or opts.get("cross_tab")) if isinstance(opts, dict) else False
        )

        if not query and not level and not class_ and not category and not tags:
            continue

        label = query or " · ".join(
            p for p in [level, class_, category, *(tags or [])] if p
        )
        await emit(job, "search_started", query=f"[keyword] {label}")
        result = await keyword_search_catalog(
            db,
            query=query,
            level=level,
            class_=class_,
            category=category,
            tags=tags,
            tag_logic=str(tag_logic),
            cross_tab=cross_tab,
            limit=DESIGNER_AGENT_SEARCH_LIMIT,
        )
        hits = result.get("hits") or []
        ids = [h.get("id") for h in hits if h.get("id") is not None]
        await emit(job, "search_done", query=f"[keyword] {label}", raw_ids=ids)

        for hit in hits:
            cid = int(hit["id"])
            if cid in seen:
                pruned.append({
                    "id": cid,
                    "name": hit.get("name"),
                    "query": label,
                    "reason": "Duplicate keyword hit.",
                })
                continue
            seen.add(cid)
            metadata = get_image_metadata(hit.get("filename") or "") or {}
            refs.append({
                "id": cid,
                "name": hit.get("name"),
                "filename": hit.get("filename"),
                "url": hit.get("url"),
                "label": hit.get("name"),
                "metadata": metadata,
                "source": "keyword",
            })
            if len(refs) >= DESIGNER_AGENT_MAX_REFERENCES:
                return refs, pruned

    return refs, pruned


async def _search_references(
    *,
    db: AsyncSession,
    collection,
    embedding_model: GeminiEmbeddingModel,
    queries: list[str],
    job: DesignerJob,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    refs: list[dict[str, Any]] = []
    pruned: list[dict[str, Any]] = []
    seen: set[int] = set()

    for query in queries:
        _check_cancelled(job)
        await emit(job, "search_started", query=query)
        _dbg("[DESIGNER ▶ EMBEDDING]", {"query": query})
        embedding = await asyncio.to_thread(embedding_model.get_text_embedding, query)
        _dbg("[DESIGNER ◀ EMBEDDING]", {"query": query, "dimensions": len(embedding)})
        result = await asyncio.to_thread(chromadb.retrieve, collection, embedding, DESIGNER_AGENT_SEARCH_LIMIT)
        ids = (result.get("ids") or [[]])[0]
        _dbg("[DESIGNER ◀ CHROMA]", {"query": query, "ids": ids})
        await emit(job, "search_done", query=query, raw_ids=ids)

        if not ids:
            continue
        candidate_filenames = _filename_candidates(ids)
        rows = (
            await db.execute(select(DBImage).where(DBImage.filename.in_(candidate_filenames)))
        ).scalars().all()
        row_by_filename: dict[str, DBImage] = {}
        for row in rows:
            for candidate in _single_filename_candidates(row.filename):
                row_by_filename[candidate] = row
            row_by_filename[row.filename] = row

        for filename in ids:
            row = row_by_filename.get(filename)
            if not row:
                pruned.append({
                    "filename": filename,
                    "query": query,
                    "reason": "No matching catalog record after filename variant lookup.",
                })
                continue
            if int(row.id) in seen:
                pruned.append({
                    "id": int(row.id),
                    "name": row.name or row.filename,
                    "filename": row.filename,
                    "query": query,
                    "reason": "Duplicate of an already selected reference.",
                })
                continue
            seen.add(int(row.id))
            metadata = get_image_metadata(row.filename) or {}
            refs.append({
                "id": int(row.id),
                "name": row.name or row.filename,
                "filename": row.filename,
                "url": row.url,
                "label": row.name or row.filename,
                "metadata": metadata,
            })
            if len(refs) >= DESIGNER_AGENT_MAX_REFERENCES:
                remaining = [rid for rid in ids if rid != filename]
                pruned.extend({
                    "filename": rid,
                    "query": query,
                    "reason": "Reference board full — lower-ranked hit skipped.",
                } for rid in remaining)
                return refs, pruned

    return refs, pruned


def _filename_candidates(filenames: list[str]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for filename in filenames:
        for candidate in _single_filename_candidates(filename):
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def _single_filename_candidates(filename: str) -> list[str]:
    base, dot, ext = filename.rpartition(".")
    if not dot:
        return [filename]
    variants = [filename]
    for new_ext in ("jpg", "jpeg", "png", "webp"):
        variants.append(f"{base}.{new_ext}")
    # Preserve order and de-duplicate.
    result: list[str] = []
    seen: set[str] = set()
    for item in variants:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


async def _evaluate_generation(
    *,
    user_image: Image.Image,
    generated_image: Image.Image,
    plan: dict[str, Any],
    references: list[dict[str, Any]],
    reference_images: list[Image.Image],
    prompt: str,
    user_id: Optional[int] = None,
) -> dict[str, Any]:
    client = get_gemini_client()
    prompt_loader.ensure_prompts_loaded()
    eval_prompt = (
        f"{prompt_loader.DESIGNER_EVALUATE_GENERATION}\n\n"
        f"Plan: {json.dumps(plan, ensure_ascii=False)}\n"
        f"References: {json.dumps(references, ensure_ascii=False)}\n"
        f"Prompt used: {prompt}"
    )
    _dbg(
        "[DESIGNER ▶ GEMINI evaluate]",
        {
            "model": DESIGNER_AGENT_MODEL,
            "user_image": _image_summary(user_image),
            "generated_image": _image_summary(generated_image),
            "reference_images": [_image_summary(img) for img in reference_images],
            "prompt_chars": len(eval_prompt),
            "prompt_preview": eval_prompt[:1200],
            "response_mime_type": "application/json",
        },
    )
    if llm_provider.use_claude():
        blocks = [
            llm_provider.pil_image_block(user_image),
            llm_provider.pil_image_block(generated_image),
        ]
        blocks += [llm_provider.pil_image_block(img) for img in reference_images]
        blocks.append(llm_provider.text_block(eval_prompt))
        text = await llm_provider.agenerate_text("", blocks) or "{}"
    else:
        response = await client.aio.models.generate_content(
            model=DESIGNER_AGENT_MODEL,
            contents=[user_image, generated_image, *reference_images, eval_prompt],
            config=GenerateContentConfig(response_mime_type="application/json"),
        )
        text = response.text or "{}"
        from app.admin.usage_helpers import record_from_response
        record_from_response(
            response,
            agent="designer",
            model=DESIGNER_AGENT_MODEL,
            user_id=user_id,
        )
        _dbg(
            "[DESIGNER ◀ GEMINI evaluate]",
            {"usage": _usage(response), "text_chars": len(text), "text_preview": text[:1200]},
        )
    try:
        data = llm_provider.parse_json_text(text) if llm_provider.use_claude() else json.loads(text)
    except (json.JSONDecodeError, ValueError):
        data = {"score": 6, "passed": False, "strengths": [], "issues": [text[:500]], "improvements": ""}
    data["passed"] = _passes_quality_gate(data)
    return data


async def _summarize_final(
    plan: dict[str, Any],
    best: dict[str, Any],
    attempts: list[dict[str, Any]],
    *,
    user_id: Optional[int] = None,
) -> str:
    client = get_gemini_client()
    prompt_loader.ensure_prompts_loaded()
    prompt = (
        f"{prompt_loader.DESIGNER_FINAL_SUMMARY}\n\n"
        f"Plan: {json.dumps(plan, ensure_ascii=False)}\n"
        f"Best: {json.dumps(best, ensure_ascii=False)}\n"
        f"Attempt count: {len(attempts)}"
    )
    _dbg(
        "[DESIGNER ▶ GEMINI final_summary]",
        {"model": DESIGNER_AGENT_MODEL, "prompt_chars": len(prompt), "prompt_preview": prompt[:1200]},
    )
    if llm_provider.use_claude():
        text = await llm_provider.agenerate_text(
            "", [llm_provider.text_block(prompt)], thinking=False,
        ) or "Your Ducon design preview is ready."
    else:
        response = await client.aio.models.generate_content(
            model=DESIGNER_AGENT_MODEL,
            contents=prompt,
        )
        text = response.text or "Your Ducon design preview is ready."
        from app.admin.usage_helpers import record_from_response
        record_from_response(
            response,
            agent="designer",
            model=DESIGNER_AGENT_MODEL,
            user_id=user_id,
        )
        _dbg(
            "[DESIGNER ◀ GEMINI final_summary]",
            {"usage": _usage(response), "text_chars": len(text), "text_preview": text[:1200]},
        )
    return text


async def _load_generation_image(db: AsyncSession, generation_id: int) -> Image.Image:
    _dbg("[DESIGNER ▶ LOAD GENERATION]", {"generation_id": generation_id})
    row = (
        await db.execute(select(Generation).where(Generation.id == generation_id))
    ).scalar_one_or_none()
    if not row:
        raise RuntimeError(f"Generation {generation_id} not found after creation.")
    if storage.CLOUD_STORAGE:
        import httpx
        url = storage.get_generation_url(row.id, row.url)
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch generated image for evaluation ({resp.status_code}).")
        # PIL decode is CPU-bound — offload so the event loop keeps streaming.
        content = resp.content
        img = await asyncio.to_thread(lambda: Image.open(io.BytesIO(content)).convert("RGB"))
        _dbg("[DESIGNER ◀ LOAD GENERATION]", {"generation_id": generation_id, "image": _image_summary(img)})
        return img
    local = storage.serve_local_path(row.url)
    if not local.exists():
        raise RuntimeError(f"Generated image file is missing: {local}")
    img = await asyncio.to_thread(lambda: Image.open(local).convert("RGB"))
    _dbg("[DESIGNER ◀ LOAD GENERATION]", {"generation_id": generation_id, "path": str(local), "image": _image_summary(img)})
    return img


async def _load_reference_images(db: AsyncSession, references: list[dict[str, Any]]) -> list[Image.Image]:
    # Resolve all reference rows in one query, then fetch each image bytes
    # concurrently. Fetches are independent (local disk or R2 HTTP), so running
    # them in parallel cuts the per-attempt reference-load time from
    # ~N*latency to ~1*latency on the designer's hot path.
    ids = [int(ref["id"]) for ref in references if ref.get("id") is not None]
    if not ids:
        return []
    rows = (
        await db.execute(select(DBImage).where(DBImage.id.in_(ids)))
    ).scalars().all()
    row_by_id = {int(r.id): r for r in rows}

    async def _load_one(ref: dict[str, Any]) -> Optional[Image.Image]:
        ref_id = ref.get("id")
        if ref_id is None:
            return None
        row = row_by_id.get(int(ref_id))
        if row is None:
            return None
        try:
            img = await load_ducon_image(row)
            # Decode + resize + re-encode is CPU-bound; offload per reference so
            # concurrent gather() actually parallelizes on the thread pool.
            return await asyncio.to_thread(lambda: normalize_user_image(_pil_to_bytes(img)))
        except Exception as exc:
            _dbg("[DESIGNER ⚠ LOAD REFERENCE FAILED]", {"id": ref_id, "error": str(exc)})
            return None

    loaded = await asyncio.gather(*[_load_one(ref) for ref in references])
    return [img for img in loaded if img is not None]


def _compose_generation_prompt(
    *,
    base_prompt: str,
    plan: dict[str, Any],
    attempt_no: int,
    previous_feedback: Optional[str],
) -> str:
    prompt_loader.ensure_prompts_loaded()
    preserve_list = ", ".join(plan.get("preserve") or []) or "important existing architecture and spatial layout"
    success_list = ", ".join(plan.get("success_criteria") or [])
    retry_block = ""
    if attempt_no > 1 and previous_feedback:
        retry_block = f"\nImprove this attempt based on reviewer feedback: {previous_feedback}\n"
    suffix = prompt_loader.render_prompt_template(
        prompt_loader.DESIGNER_COMPOSE_GENERATION_SUFFIX,
        PRESERVE_LIST=preserve_list,
        SUCCESS_CRITERIA_LIST=success_list,
        RETRY_FEEDBACK_BLOCK=retry_block,
    )
    prompt = f"{base_prompt}\n\n{suffix}".strip()
    return prompt


def _fallback_queries(user_prompt: str) -> list[str]:
    base = user_prompt or "modern luxury outdoor living design"
    return [
        base,
        "premium outdoor terrace landscaping",
        "modern pool patio pergola Ducon design",
        "luxury villa outdoor seating pavers",
    ]


def _fallback_generation_prompt(user_prompt: str) -> str:
    return (
        user_prompt
        or "Redesign the client's outdoor space as a premium Ducon concept with refined paving, "
        "balanced greenery, elegant lighting, and a realistic luxury villa exterior atmosphere."
    )


def _check_cancelled(job: DesignerJob) -> None:
    if job.cancel_requested:
        raise asyncio.CancelledError()
