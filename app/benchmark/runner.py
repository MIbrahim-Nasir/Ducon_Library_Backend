"""Benchmark case runner.

Exposes a single coroutine, :func:`run_benchmark_case`, that the dev router
(``app/routers/dev_benchmark.py`` — owned by another worker) calls to execute
one benchmark case end-to-end against the live image-gen pipeline with
per-call config overrides and dry-run semantics (no DB / storage / disk
writes).

Concurrency: ``run_benchmark_case`` is safe to call many times concurrently
via ``asyncio.gather``. Per-call overrides are threaded explicitly down the
call chain; we never mutate global ``cfg()`` state.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from typing import Awaitable, Callable, Optional

from PIL import Image

from app.benchmark.types import (
    BenchmarkConfig,
    BenchmarkInput,
    BenchmarkResult,
    BenchmarkStep,
    RunOverrides,
    StepMetrics,
)
from app.gemini import (
    combine_images,
    generate_image,
)
from app.image_gen_agent import ImageGenAgent

logger = logging.getLogger(__name__)

# A minimal default prompt for the ``direct`` flow when no
# ``system_prompt_override`` is supplied on the config.
_DIRECT_DEFAULT_PROMPT = (
    "Photorealistic architectural visualization. Apply the Ducon design language "
    "from the reference image into the user's outdoor space. Preserve all existing "
    "fixed architecture, camera angle, and site geometry. No hallucinated structures."
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _png_bytes(pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _inputs_summary(inputs: list[BenchmarkInput]) -> list[dict]:
    return [
        {"label": inp.label, "role": inp.role, "has_metadata": inp.metadata is not None}
        for inp in inputs
    ]


def _build_cost_breakdown(steps: list[BenchmarkStep]) -> list[dict]:
    """Aggregate per-model cost across all completed steps."""
    by_model: dict[str, dict] = {}
    for s in steps:
        if s.status != "completed":
            continue
        key = s.model or "unknown"
        entry = by_model.setdefault(
            key,
            {"model": key, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "calls": 0},
        )
        if s.cost_usd is not None:
            entry["cost_usd"] += s.cost_usd
        if s.tokens_in is not None:
            entry["tokens_in"] += s.tokens_in
        if s.tokens_out is not None:
            entry["tokens_out"] += s.tokens_out
        entry["calls"] += 1
    for entry in by_model.values():
        entry["cost_usd"] = round(entry["cost_usd"], 6)
    return list(by_model.values())


# ── Public API ────────────────────────────────────────────────────────────────


async def run_benchmark_case(
    *,
    process_id: str,
    config: BenchmarkConfig,
    inputs: list[BenchmarkInput],
    hint: Optional[str],
    use_ducon_data: bool,
    on_step: Optional[Callable[[BenchmarkStep], Awaitable[None]]] = None,
    run_group_id: Optional[str] = None,
) -> BenchmarkResult:
    """Run one benchmark case and return a :class:`BenchmarkResult`.

    The run is wrapped in try/except so a failure produces a
    ``status="failed"`` result rather than raising — the router runs many
    cases concurrently and a single failure must not crash the worker.
    """
    t_start = time.perf_counter()
    steps: list[BenchmarkStep] = []
    step_index = 0
    overrides: RunOverrides = config.to_overrides()

    # ── on_step fan-out that also accumulates into our local steps list ──────
    async def _emit(step: BenchmarkStep) -> None:
        # The instrumented pipeline already records steps internally; here we
        # collect them and forward to the caller's callback. Running/completed
        # pairs share the same index — update in place instead of duplicating.
        for i, existing in enumerate(steps):
            if existing.index == step.index and existing.kind == step.kind:
                steps[i] = step
                break
        else:
            steps.append(step)
        if on_step is not None:
            try:
                await on_step(step)
            except Exception as cb_exc:
                logger.warning(
                    "[benchmark %s] on_step callback failed: %s", process_id, cb_exc
                )

    # For flows where the pipeline does not emit its own steps (agent_loop,
    # direct), we build steps here via a small helper.
    async def _emit_local(
        kind: str,
        model: str,
        thinking: Optional[str],
        status: str,
        started_at: float,
        ended_at: Optional[float],
        metrics: Optional[StepMetrics] = None,
        *,
        prompt_used: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        nonlocal step_index
        dur = int((ended_at - started_at) * 1000) if ended_at else None
        step = BenchmarkStep(
            index=step_index,
            kind=kind,
            model=model or "",
            thinking=thinking,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=dur,
            prompt_used=prompt_used,
            tokens_in=metrics.tokens_in if metrics else None,
            tokens_out=metrics.tokens_out if metrics else None,
            image_count=metrics.image_count if metrics else None,
            cost_usd=metrics.cost_usd if metrics else None,
            error=error,
        )
        if status != "running":
            step_index += 1
        await _emit(step)

    output_images: list[bytes] = []
    final_prompt: Optional[str] = None
    retries = 0
    total_cost = 0.0
    status = "completed"
    error: Optional[str] = None

    try:
        if config.flow == "agent_loop":
            final_prompt, output_images, retries = await _run_agent_loop(
                config=config,
                inputs=inputs,
                hint=hint,
                use_ducon_data=use_ducon_data,
                overrides=overrides,
                emit_local=_emit_local,
            )
        elif config.flow == "direct":
            final_prompt, output_images = await _run_direct(
                config=config,
                inputs=inputs,
                overrides=overrides,
                emit_local=_emit_local,
            )
        elif config.flow == "multi_image":
            final_prompt, output_images, retries = await _run_multi_image(
                config=config,
                inputs=inputs,
                hint=hint,
                overrides=overrides,
                emit=_emit,
            )
        elif config.flow == "designer":
            if not run_group_id:
                raise ValueError("designer flow requires run_group_id")
            final_prompt, output_images, retries = await _run_designer(
                config=config,
                inputs=inputs,
                hint=hint,
                overrides=overrides,
                process_id=process_id,
                run_group_id=run_group_id,
                emit=_emit,
            )
        else:
            raise ValueError(f"Unknown benchmark flow: {config.flow!r}")
    except Exception as exc:
        logger.exception("[benchmark %s] run failed: %s", process_id, exc)
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"

    # ── Assemble result ──────────────────────────────────────────────────────
    total_duration_ms = int((time.perf_counter() - t_start) * 1000)
    cost_breakdown = _build_cost_breakdown(steps)
    total_cost = round(sum(e["cost_usd"] for e in cost_breakdown), 6)

    if not retries:
        retries = sum(
            1 for s in steps if s.kind == "prompt_retry" and s.status == "completed"
        )

    return BenchmarkResult(
        process_id=process_id,
        config=config,
        inputs_summary=_inputs_summary(inputs),
        steps=steps,
        output_images=output_images,
        final_prompt=final_prompt,
        retries=retries,
        total_duration_ms=total_duration_ms,
        total_cost_usd=total_cost,
        cost_breakdown=cost_breakdown,
        status=status,
        error=error,
    )


# ── Flow: agent_loop (2-image autogenerate-style) ─────────────────────────────


async def _run_agent_loop(
    *,
    config: BenchmarkConfig,
    inputs: list[BenchmarkInput],
    hint: Optional[str],
    use_ducon_data: bool,
    overrides: RunOverrides,
    emit_local: Callable,
) -> tuple[Optional[str], list[bytes], int]:
    if not inputs:
        raise ValueError("agent_loop requires at least one input image.")

    # Identify the user-space input and the primary design reference.
    user_input = next((i for i in inputs if i.role == "user"), None)
    if user_input is None:
        # Fall back to the last input as the user space.
        user_input = inputs[-1]
    design_input = next((i for i in inputs if i.role in ("design", "product", "area")), None)
    if design_input is None:
        # Single-input case: treat the sole input as the design reference and
        # mark image2 as user space with no second image.
        design_input = inputs[0]

    image2_is_user_space = user_input is not design_input
    image1 = design_input.pil_image
    image2 = user_input.pil_image if image2_is_user_space else None
    if image2 is None:
        # combine_images expects two images; reuse image1 as a placeholder
        # second slot only when there is genuinely one input.
        image2 = image1

    image1_metadata = design_input.metadata if use_ducon_data else None
    image2_metadata = user_input.metadata if (use_ducon_data and image2_is_user_space) else None

    agent = ImageGenAgent(
        image1_name=design_input.label,
        image2_name=user_input.label if image2_is_user_space else None,
        image2_is_user_space=image2_is_user_space,
        prompt_model=overrides.prompt_model,
        prompt_thinking=overrides.prompt_thinking,
        max_eval_rounds=overrides.max_eval_rounds,
    )

    # ── Step 1: initial prompt ───────────────────────────────────────────────
    t0 = time.perf_counter()
    sm = StepMetrics()
    await emit_local("prompt_initial", config.prompt_model, config.prompt_thinking,
                     "running", t0, None, None)
    try:
        active_prompt = await agent.generate_initial_prompt(
            image1=image1,
            image2=image2,
            image1_metadata=image1_metadata,
            image2_metadata=image2_metadata,
            user_hint=hint,
            metrics=sm,
        )
        await emit_local("prompt_initial", config.prompt_model, config.prompt_thinking,
                         "completed", t0, time.perf_counter(), sm,
                         prompt_used=active_prompt)
    except Exception as exc:
        await emit_local("prompt_initial", config.prompt_model, config.prompt_thinking,
                         "failed", t0, time.perf_counter(), sm, error=str(exc))
        raise

    final_prompt = active_prompt
    output_images: list[bytes] = []
    max_rounds = int(overrides.max_eval_rounds) if overrides.max_eval_rounds else 3
    retries = 0

    # ── Step 2: generation + eval loop ───────────────────────────────────────
    for gen_round in range(max_rounds):
        # image_gen
        t0 = time.perf_counter()
        gsm = StepMetrics()
        await emit_local("image_gen", config.image_model, config.image_thinking,
                         "running", t0, None, None)
        try:
            png_bytes = await asyncio.to_thread(
                combine_images,
                f"bench_agentloop_{gen_round}.png",  # filename unused in dry_run
                image1,
                image2,
                active_prompt,
                None,  # subfolder
                config.aspect_ratio,
                image_model=overrides.image_model,
                image_thinking=overrides.image_thinking,
                dry_run=True,
                metrics=gsm,
            )
            output_images.append(png_bytes)
            await emit_local("image_gen", config.image_model, config.image_thinking,
                             "completed", t0, time.perf_counter(), gsm,
                             prompt_used=active_prompt)
        except Exception as exc:
            await emit_local("image_gen", config.image_model, config.image_thinking,
                             "failed", t0, time.perf_counter(), gsm, error=str(exc))
            raise

        # eval
        t0 = time.perf_counter()
        esm = StepMetrics()
        await emit_local("eval", config.prompt_model, config.prompt_thinking,
                         "running", t0, None, None)
        try:
            generated_pil = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            approved, issues = await agent.evaluate_output(
                generated=generated_pil,
                gen_round=gen_round + 1,
                metrics=esm,
            )
            await emit_local("eval", config.prompt_model, config.prompt_thinking,
                             "completed", t0, time.perf_counter(), esm)
        except Exception as exc:
            await emit_local("eval", config.prompt_model, config.prompt_thinking,
                             "failed", t0, time.perf_counter(), esm, error=str(exc))
            raise

        if approved:
            break

        if gen_round + 1 >= max_rounds:
            break

        # prompt_retry
        t0 = time.perf_counter()
        rsm = StepMetrics()
        await emit_local("prompt_retry", config.prompt_model, config.prompt_thinking,
                         "running", t0, None, None)
        try:
            active_prompt = await agent.generate_retry_prompt(
                gen_round=gen_round + 1,
                issues=issues or [],
                metrics=rsm,
            )
            final_prompt = active_prompt
            retries += 1
            await emit_local("prompt_retry", config.prompt_model, config.prompt_thinking,
                             "completed", t0, time.perf_counter(), rsm,
                             prompt_used=active_prompt)
        except Exception as exc:
            await emit_local("prompt_retry", config.prompt_model, config.prompt_thinking,
                             "failed", t0, time.perf_counter(), rsm, error=str(exc))
            raise

    return final_prompt, output_images, retries


# ── Flow: direct (no agent, no eval loop) ─────────────────────────────────────


async def _run_direct(
    *,
    config: BenchmarkConfig,
    inputs: list[BenchmarkInput],
    overrides: RunOverrides,
    emit_local: Callable,
) -> tuple[Optional[str], list[bytes]]:
    prompt_text = config.system_prompt_override or _DIRECT_DEFAULT_PROMPT
    final_prompt = prompt_text
    output_images: list[bytes] = []

    t0 = time.perf_counter()
    sm = StepMetrics()
    await emit_local("direct_gen", config.image_model, config.image_thinking,
                     "running", t0, None, None)
    try:
        if len(inputs) >= 2:
            image1 = inputs[0].pil_image
            image2 = inputs[1].pil_image
            png_bytes = await asyncio.to_thread(
                combine_images,
                "bench_direct.png",
                image1,
                image2,
                prompt_text,
                None,
                config.aspect_ratio,
                image_model=overrides.image_model,
                image_thinking=overrides.image_thinking,
                dry_run=True,
                metrics=sm,
            )
        elif len(inputs) == 1:
            # Single reference image: still use combine_images with the image
            # duplicated so the model has the expected two-image context.
            image1 = inputs[0].pil_image
            png_bytes = await asyncio.to_thread(
                combine_images,
                "bench_direct.png",
                image1,
                image1,
                prompt_text,
                None,
                config.aspect_ratio,
                image_model=overrides.image_model,
                image_thinking=overrides.image_thinking,
                dry_run=True,
                metrics=sm,
            )
        else:
            # Text-only: standalone text-to-image.
            png_bytes = await asyncio.to_thread(
                generate_image,
                prompt_text,
                image_model=overrides.image_model,
                image_thinking=overrides.image_thinking,
                aspect_ratio=config.aspect_ratio,
                dry_run=True,
                metrics=sm,
            )
        output_images.append(png_bytes)
        await emit_local("direct_gen", config.image_model, config.image_thinking,
                         "completed", t0, time.perf_counter(), sm,
                         prompt_used=prompt_text)
    except Exception as exc:
        await emit_local("direct_gen", config.image_model, config.image_thinking,
                         "failed", t0, time.perf_counter(), sm, error=str(exc))
        raise

    return final_prompt, output_images


# ── Flow: multi_image (delegate to generate_multi_image dry-run) ─────────────


async def _run_multi_image(
    *,
    config: BenchmarkConfig,
    inputs: list[BenchmarkInput],
    hint: Optional[str],
    overrides: RunOverrides,
    emit: Callable,
) -> tuple[Optional[str], list[bytes], int]:
    from app.tool_generate_image import ImageDescriptor, generate_multi_image

    # Build file-type descriptors carrying PIL images so dry-run needs no DB.
    descriptors = [
        ImageDescriptor(
            label=inp.label,
            type="file",
            source=None,
            pil_image=inp.pil_image,
        )
        for inp in inputs
    ]

    # generate_multi_image emits its own BenchmarkStep objects via on_step; we
    # forward them through ``emit`` so they reach the caller and our local list.
    result = await generate_multi_image(
        user_id=None,
        guest_session_id=None,
        prompt=hint or "",
        descriptors=descriptors,
        model="pro",  # ignored — image_model override is supplied below
        aspect_ratio=config.aspect_ratio,
        db=None,
        enable_verify=True,
        image_model=overrides.image_model,
        image_thinking=overrides.image_thinking,
        prompt_model=overrides.prompt_model,
        prompt_thinking=overrides.prompt_thinking,
        max_eval_rounds=overrides.max_eval_rounds,
        max_prompt_verify_rounds=overrides.max_prompt_verify_rounds,
        dry_run=True,
        on_step=emit,
    )

    pil_images = result.get("pil_images") or []
    output_images = [_png_bytes(p) for p in pil_images]
    # If pil_images weren't returned for some reason, fall back to raw bytes.
    if not output_images and result.get("image_bytes"):
        output_images = [result["image_bytes"]]

    final_prompt = result.get("final_prompt") or hint
    retries = int(result.get("retries") or 0)
    return final_prompt, output_images, retries


# ── Flow: designer (dev designer agent pipeline) ─────────────────────────────


async def _run_designer(
    *,
    config: BenchmarkConfig,
    inputs: list[BenchmarkInput],
    hint: Optional[str],
    overrides: RunOverrides,
    process_id: str,
    run_group_id: str,
    emit: Callable[[BenchmarkStep], Awaitable[None]],
) -> tuple[Optional[str], list[bytes], int]:
    from app.benchmark import designer_agent
    from app.benchmark.provider_registry import ModelRouterPair

    if not inputs:
        raise ValueError("designer flow requires at least one input image.")

    user_input = next((i for i in inputs if i.role == "user"), inputs[0])
    buf = io.BytesIO()
    user_input.pil_image.save(buf, format="PNG")
    user_image_bytes = buf.getvalue()

    prompt_pair = config.prompt_model_pair or {
        "router": config.prompt_router,
        "model_id": config.prompt_model,
    }
    image_pair = config.image_model_pair or {
        "router": config.image_router,
        "model_id": config.image_model,
    }
    model_pair = ModelRouterPair(
        id=str(prompt_pair.get("id") or f"{prompt_pair.get('router')}:{prompt_pair.get('model_id')}"),
        router=str(prompt_pair.get("router") or "gemini_native"),
        model_id=str(prompt_pair.get("model_id") or config.prompt_model),
        label=str(prompt_pair.get("model_id") or config.prompt_model),
        roles=["designer", "prompt"],
        thinking_modes=[],
    )
    image_model_pair = ModelRouterPair(
        id=str(image_pair.get("id") or f"{image_pair.get('router')}:{image_pair.get('model_id')}"),
        router=str(image_pair.get("router") or "gemini_native"),
        model_id=str(image_pair.get("model_id") or config.image_model),
        label=str(image_pair.get("model_id") or config.image_model),
        roles=["image"],
        thinking_modes=[],
    )

    tool_access = set(config.designer_tool_access or designer_agent.DEFAULT_DESIGNER_TOOLS)
    mode = (config.designer_system_prompt_mode or "append").strip().lower()
    if mode not in {"append", "replace", "compose"}:
        mode = "append"

    step_index = 0
    open_steps: dict[str, tuple[int, float]] = {}

    async def on_event(payload: dict) -> None:
        nonlocal step_index
        ev_type = str(payload.get("type") or "")
        t0 = time.perf_counter()
        kind = {
            "plan": "designer_plan",
            "search_started": "designer_search",
            "search_done": "designer_search",
            "sources": "designer_sources",
            "get_image": "designer_get_image",
            "generation_started": "image_gen",
            "generation_done": "image_gen",
            "generation_round_started": "designer_generation_round",
            "eval": "eval",
            "status": "designer_status",
            "input_image": "designer_input",
        }.get(ev_type, f"designer_{ev_type}")

        if ev_type == "assistant_message":
            ended = time.perf_counter()
            await emit(
                BenchmarkStep(
                    index=step_index,
                    kind="designer_message",
                    model=config.prompt_model,
                    thinking=config.prompt_thinking or None,
                    status="completed",
                    started_at=t0,
                    ended_at=ended,
                    duration_ms=int((ended - t0) * 1000),
                    prompt_used=str(payload.get("text") or "")[:4000],
                )
            )
            step_index += 1
            return

        if ev_type == "tool_call":
            key = f"tool:{payload.get('id')}"
            open_steps[key] = (step_index, t0)
            tool_name = str(payload.get("name") or "tool")
            await emit(
                BenchmarkStep(
                    index=step_index,
                    kind=("image_gen" if tool_name == "generate_multi_image" else f"designer_{tool_name}"),
                    model=config.prompt_model,
                    thinking=config.prompt_thinking or None,
                    status="running",
                    started_at=t0,
                    prompt_used=json.dumps(payload.get("args") or {}, ensure_ascii=False)[:2000],
                )
            )
            step_index += 1
            return

        if ev_type == "tool_result":
            key = f"tool:{payload.get('id')}"
            idx, started = open_steps.pop(key, (step_index, t0))
            ended = time.perf_counter()
            tool_name = str(payload.get("name") or "tool")
            result = payload.get("result")
            failed = isinstance(result, dict) and bool(result.get("error"))
            await emit(
                BenchmarkStep(
                    index=idx,
                    kind=("image_gen" if tool_name == "generate_multi_image" else f"designer_{tool_name}"),
                    model=config.prompt_model,
                    thinking=config.prompt_thinking or None,
                    status="failed" if failed else "completed",
                    started_at=started,
                    ended_at=ended,
                    duration_ms=int((ended - started) * 1000),
                    image_count=len(payload.get("output_images") or []),
                    error=(str(result.get("error")) if failed else None),
                )
            )
            step_index = max(step_index, idx + 1)
            return

        if ev_type == "search_started":
            key = f"{payload.get('search_type')}:{payload.get('query')}"
            open_steps[key] = (step_index, t0)
            await emit(
                BenchmarkStep(
                    index=step_index,
                    kind=kind,
                    model=config.prompt_model,
                    thinking=config.prompt_thinking or None,
                    status="running",
                    started_at=t0,
                    prompt_used=str(payload.get("query") or ""),
                )
            )
            return

        if ev_type == "search_done":
            key = f"{payload.get('search_type')}:{payload.get('query')}"
            idx, started = open_steps.pop(key, (step_index, t0))
            ended = time.perf_counter()
            await emit(
                BenchmarkStep(
                    index=idx,
                    kind=kind,
                    model=config.prompt_model,
                    thinking=config.prompt_thinking or None,
                    status="completed",
                    started_at=started,
                    ended_at=ended,
                    duration_ms=int((ended - started) * 1000),
                    prompt_used=str(payload.get("query") or ""),
                )
            )
            step_index = max(step_index, idx + 1)
            return

        if ev_type == "generation_started":
            open_steps["generation_started"] = (step_index, t0)
            await emit(
                BenchmarkStep(
                    index=step_index,
                    kind=kind,
                    model=config.image_model,
                    thinking=config.image_thinking or None,
                    status="running",
                    started_at=t0,
                )
            )
            return

        if ev_type == "generation_done":
            idx, started = open_steps.pop("generation_started", (step_index, t0))
            ended = time.perf_counter()
            await emit(
                BenchmarkStep(
                    index=idx,
                    kind=kind,
                    model=config.image_model,
                    thinking=config.image_thinking or None,
                    status="completed",
                    started_at=started,
                    ended_at=ended,
                    duration_ms=int((ended - started) * 1000),
                    image_count=len(payload.get("output_images") or []),
                )
            )
            step_index = max(step_index, idx + 1)
            return

        if ev_type == "eval":
            # Per-round self-evaluation produced by the dev designer loop. The
            # designer emits a single ``eval`` event after each evaluation, so we
            # record it as one completed step (no running/started pair).
            ended = time.perf_counter()
            approved = bool(payload.get("approved"))
            defects = payload.get("defects") or []
            verdict = "approved" if approved else ("rejected: " + "; ".join(str(d) for d in defects))
            await emit(
                BenchmarkStep(
                    index=step_index,
                    kind="eval",
                    model=config.prompt_model,
                    thinking=config.prompt_thinking or None,
                    status="completed",
                    started_at=t0,
                    ended_at=ended,
                    duration_ms=int((ended - t0) * 1000),
                    prompt_used=verdict,
                )
            )
            step_index += 1
            return

        if ev_type == "plan":
            import json as _json

            ended = time.perf_counter()
            plan = payload.get("plan")
            prompt_used = _json.dumps(plan, ensure_ascii=False)[:4000] if plan else None
            await emit(
                BenchmarkStep(
                    index=step_index,
                    kind=kind,
                    model=config.prompt_model,
                    thinking=config.prompt_thinking or None,
                    status="completed",
                    started_at=t0,
                    ended_at=ended,
                    duration_ms=int((ended - t0) * 1000),
                    prompt_used=prompt_used,
                )
            )
            step_index += 1
            return

        if ev_type in {"sources", "get_image", "status", "input_image"}:
            ended = time.perf_counter()
            await emit(
                BenchmarkStep(
                    index=step_index,
                    kind=kind,
                    model=config.prompt_model,
                    thinking=config.prompt_thinking or None,
                    status="completed",
                    started_at=t0,
                    ended_at=ended,
                    duration_ms=int((ended - t0) * 1000),
                )
            )
            step_index += 1

    _dmt = getattr(config, "designer_max_turns", None)
    _dwc = getattr(config, "designer_wall_clock_budget_s", None)
    final_prompt, output_images, _final = await designer_agent.run_designer_benchmark(
        process_id=process_id,
        run_group_id=run_group_id,
        user_image_bytes=user_image_bytes,
        user_prompt=hint or "",
        model_pair=model_pair,
        image_pair=image_model_pair,
        thinking=overrides.prompt_thinking,
        image_thinking=overrides.image_thinking,
        system_prompt=config.system_prompt_override,
        system_prompt_mode=mode,
        system_prompt_sections=None,
        tool_access=tool_access,
        filesystem_root=config.designer_filesystem_root,
        aspect_ratio=config.aspect_ratio,
        max_generation_rounds=getattr(config, "designer_max_generation_rounds", None)
        or designer_agent.DEFAULT_MAX_GENERATION_ROUNDS,
        max_turns=_dmt if _dmt is not None else designer_agent.DEFAULT_MAX_TURNS,
        wall_clock_budget_s=_dwc,
        on_event=on_event,
    )
    return final_prompt, output_images, 0
