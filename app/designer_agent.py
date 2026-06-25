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
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import chromadb, storage
from app.db.models import Generation, Image as DBImage, User
from app.gemini import (
    get_gemini_client,
    verify_prompt as _verify_prompt,
    PROMPT_VERIFY_MAX_ROUNDS,
)
from app.image_utils import get_image_metadata, normalize_user_image
from app.image_utils import load_ducon_image
from app.ml import GeminiEmbeddingModel
from app.tool_generate_image import ImageDescriptor, generate_multi_image
from app.prompt_generator_session import DesignerPromptSession
from app import prompt_loader


DESIGNER_AGENT_MODEL = os.getenv("DESIGNER_AGENT_MODEL", "gemini-3.5-flash")
DESIGNER_AGENT_MAX_STEPS = int(os.getenv("DESIGNER_AGENT_MAX_STEPS", "12"))
DESIGNER_AGENT_MAX_GENERATIONS = int(os.getenv("DESIGNER_AGENT_MAX_GENERATIONS", "3"))
DESIGNER_AGENT_SEARCH_LIMIT = int(os.getenv("DESIGNER_AGENT_SEARCH_LIMIT", "5"))
DESIGNER_AGENT_MAX_REFERENCES = int(os.getenv("DESIGNER_AGENT_MAX_REFERENCES", "6"))
DESIGNER_AGENT_PASS_SCORE = float(os.getenv("DESIGNER_AGENT_PASS_SCORE", "7.5"))
DESIGNER_AGENT_DEFAULT_IMAGE_MODEL = os.getenv("DESIGNER_AGENT_IMAGE_MODEL", "flash")
DESIGNER_AGENT_DEFAULT_ASPECT_RATIO = os.getenv("DESIGNER_AGENT_ASPECT_RATIO", "16:9")
_LIVE_DEBUG: bool = os.getenv("LIVE_DEBUG", "").lower() in ("1", "true", "yes")

_CRITICAL_SECTION_KEYS = (
    "A1_pov",
    "A2_structures",
    "A3_scene",
    "B1_area_products",
    "B2_fixed_products",
    "B3_zones",
    "C1_no_extra",
    "C2_no_missing",
)


def _enrich_generation_record(gen: dict[str, Any]) -> dict[str, Any]:
    """Return generation dict with a fresh presigned URL for the UI."""
    enriched = dict(gen)
    gen_id = enriched.get("id")
    if gen_id is not None:
        enriched["signed_url"] = storage.get_generation_url(int(gen_id), enriched.get("url"))
    return enriched


def _dbg(*args) -> None:
    if _LIVE_DEBUG:
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


def create_job(user: User) -> DesignerJob:
    job = DesignerJob(id=uuid.uuid4().hex, user_id=int(user.id))
    JOBS[job.id] = job
    return job


async def stream_job_events(job: DesignerJob):
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


def cancel_job(job_id: str, user_id: int) -> bool:
    job = JOBS.get(job_id)
    if not job or job.user_id != user_id:
        return False
    job.cancel_requested = True
    if job.task and not job.task.done():
        job.task.cancel()
    return True


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

        user_image = normalize_user_image(user_image_bytes)
        _dbg("[DESIGNER ▶ INPUT IMAGE]", _image_summary(user_image))
        user_prompt = (user_prompt or "").strip()

        plan = await _analyze_and_plan(user_image, user_prompt)
        await emit(job, "plan", plan=plan)

        search_queries = plan.get("search_queries") or []
        if not search_queries:
            search_queries = _fallback_queries(user_prompt)

        references, pruned_references = await _search_references(
            db=db,
            collection=collection,
            embedding_model=embedding_model,
            queries=search_queries[:4],
            job=job,
        )
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

        final_summary = await _summarize_final(plan=plan, best=best, attempts=attempts)
        _dbg("[DESIGNER ◀ FINAL SUMMARY]", final_summary)
        # UI shows the last generated candidate (final attempt), not highest-scored.
        last_generation = _enrich_generation_record(dict(attempts[-1]["generation"]))
        job.status = "completed"
        job.final = {
            "job_id": job.id,
            "best_generation": last_generation,
            "best_scored_generation": _enrich_generation_record(dict(best["generation"])),
            "best_evaluation": best["evaluation"],
            "references": best["references"],
            "attempts": attempts,
            "summary": final_summary,
        }
        await emit(job, "final", **job.final)

    except asyncio.CancelledError:
        job.status = "cancelled"
        await emit(job, "cancelled", message="Designer job was cancelled.")
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        _dbg("[DESIGNER ✖ ERROR]", repr(exc))
        await emit(job, "error", message=str(exc))
    finally:
        await job.queue.put(None)


async def _analyze_and_plan(user_image: Image.Image, user_prompt: str) -> dict[str, Any]:
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
    response = await client.aio.models.generate_content(
        model=DESIGNER_AGENT_MODEL,
        contents=[user_image, prompt],
        config=GenerateContentConfig(response_mime_type="application/json"),
    )
    text = response.text or "{}"
    _dbg(
        "[DESIGNER ◀ GEMINI analyze_plan]",
        {"usage": _usage(response), "text_chars": len(text), "text_preview": text[:1200]},
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "space_analysis": text[:1000],
            "design_direction": user_prompt or "Create a refined Ducon outdoor concept.",
            "preserve": [],
            "opportunities": [],
            "search_queries": _fallback_queries(user_prompt),
            "generation_prompt": _fallback_generation_prompt(user_prompt),
            "success_criteria": ["photorealistic", "fits the existing space", "uses Ducon references clearly"],
        }


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
        result = chromadb.retrieve(collection, embedding, DESIGNER_AGENT_SEARCH_LIMIT)
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
    response = await client.aio.models.generate_content(
        model=DESIGNER_AGENT_MODEL,
        contents=[user_image, generated_image, *reference_images, eval_prompt],
        config=GenerateContentConfig(response_mime_type="application/json"),
    )
    text = response.text or "{}"
    _dbg(
        "[DESIGNER ◀ GEMINI evaluate]",
        {"usage": _usage(response), "text_chars": len(text), "text_preview": text[:1200]},
    )
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {"score": 6, "passed": False, "strengths": [], "issues": [text[:500]], "improvements": ""}
    data["passed"] = _passes_quality_gate(data)
    return data


async def _summarize_final(plan: dict[str, Any], best: dict[str, Any], attempts: list[dict[str, Any]]) -> str:
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
    response = await client.aio.models.generate_content(
        model=DESIGNER_AGENT_MODEL,
        contents=prompt,
    )
    text = response.text or "Your Ducon design preview is ready."
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
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        _dbg("[DESIGNER ◀ LOAD GENERATION]", {"generation_id": generation_id, "image": _image_summary(img)})
        return img
    local = storage.serve_local_path(row.url)
    if not local.exists():
        raise RuntimeError(f"Generated image file is missing: {local}")
    img = Image.open(local).convert("RGB")
    _dbg("[DESIGNER ◀ LOAD GENERATION]", {"generation_id": generation_id, "path": str(local), "image": _image_summary(img)})
    return img


async def _load_reference_images(db: AsyncSession, references: list[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for ref in references:
        ref_id = ref.get("id")
        if ref_id is None:
            continue
        row = (
            await db.execute(select(DBImage).where(DBImage.id == int(ref_id)))
        ).scalar_one_or_none()
        if not row:
            continue
        try:
            img = await load_ducon_image(row)
            images.append(normalize_user_image(_pil_to_bytes(img)))
        except Exception as exc:
            _dbg("[DESIGNER ⚠ LOAD REFERENCE FAILED]", {"id": ref_id, "error": str(exc)})
    return images


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
