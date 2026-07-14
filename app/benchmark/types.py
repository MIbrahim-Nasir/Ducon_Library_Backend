"""Data types for the internal AI-generation benchmark dashboard.

These dataclasses are the contract between the pipeline/runner (this package),
the FastAPI dev router (``app/routers/dev_benchmark.py`` — owned by another
worker), and the frontend (owned by a third worker).

Keep this module dependency-free at import time (only stdlib + PIL) so it can
be imported by the router and tests without booting the whole app.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PIL import Image


# ── Per-call overrides ───────────────────────────────────────────────────────
# A small, focused override carrier. We accept either a full BenchmarkConfig
# (preferred, used by the runner) or this lighter struct at the function level
# so individual call sites can override a couple of values without constructing
# a whole BenchmarkConfig.


@dataclass
class RunOverrides:
    """Per-call model/thinking overrides threaded through the gen pipeline.

    Every field defaults to ``None`` meaning "use the live ``cfg()`` value".
    Passing explicit values bypasses global config reads so concurrent
    benchmark runs do not race on shared mutable config state.
    """

    image_model: Optional[str] = None
    image_router: Optional[str] = None
    image_thinking: Optional[str] = None
    prompt_model: Optional[str] = None
    prompt_router: Optional[str] = None
    prompt_thinking: Optional[str] = None
    max_eval_rounds: Optional[int] = None
    max_prompt_verify_rounds: Optional[int] = None
    system_prompt_override: Optional[str] = None
    aspect_ratio: Optional[str] = None


# ── Step-level metrics accumulator ───────────────────────────────────────────


@dataclass
class StepMetrics:
    """Mutable accumulator populated by instrumented model calls.

    Passed (as ``metrics=``) into gemini/agent functions when running under a
    benchmark. After the call the runner reads the accumulated tokens/cost to
    populate the corresponding ``BenchmarkStep``. Presence of a ``metrics``
    argument is also the signal that the call should record usage via the
    usage recorder (filling the gap where prompt/eval calls were previously
    unrecorded).
    """

    tokens_in: int = 0
    tokens_out: int = 0
    image_count: int = 0
    cost_usd: float = 0.0
    model: str = ""
    calls: int = 0

    def add(self, *, model: str, tokens_in: int, tokens_out: int,
            image_count: int = 0, cost_usd: float = 0.0) -> None:
        self.tokens_in += int(tokens_in or 0)
        self.tokens_out += int(tokens_out or 0)
        self.image_count += int(image_count or 0)
        self.cost_usd += float(cost_usd or 0.0)
        self.model = model or self.model
        self.calls += 1


# ── Public benchmark dataclasses ─────────────────────────────────────────────


@dataclass(frozen=True)
class BenchmarkConfig:
    """A single benchmark case configuration.

    ``flow`` selects which generation pipeline to run:
    - ``"agent_loop"``: 2-image autogenerate-style flow using ``ImageGenAgent``
      with prompt → image → eval → retry loop.
    - ``"direct"``: skip the agent; feed ``system_prompt_override`` (or a
      minimal default) straight to the image model.
    - ``"multi_image"``: delegate to ``generate_multi_image`` (dry-run).
    """

    flow: str
    image_model: str
    image_thinking: str  # "minimal"|"low"|"medium"|"High"|"max"
    prompt_model: str
    prompt_thinking: str
    max_eval_rounds: int
    max_prompt_verify_rounds: int
    system_prompt_override: Optional[str]
    aspect_ratio: Optional[str]
    name: str = ""
    image_router: str = "gemini_native"
    prompt_router: str = "gemini_native"
    image_model_pair: Optional[dict] = None
    prompt_model_pair: Optional[dict] = None
    designer_tool_access: Optional[tuple[str, ...]] = None
    designer_system_prompt_mode: Optional[str] = None
    designer_filesystem_root: Optional[str] = None
    # Dev designer agentic loop budgets: generation cap and model-turn cap.
    # These are guards, not scripts — the agent decides when to stop within them.
    designer_max_generation_rounds: int = 5
    designer_max_turns: int = 24
    # 0 = unlimited wall-clock budget for the designer agent loop
    designer_wall_clock_budget_s: int = 15 * 60

    def to_overrides(self) -> RunOverrides:
        return RunOverrides(
            image_model=self.image_model or None,
            image_router=self.image_router or None,
            image_thinking=self.image_thinking or None,
            prompt_model=self.prompt_model or None,
            prompt_router=self.prompt_router or None,
            prompt_thinking=self.prompt_thinking or None,
            max_eval_rounds=self.max_eval_rounds if self.max_eval_rounds else None,
            max_prompt_verify_rounds=(
                self.max_prompt_verify_rounds if self.max_prompt_verify_rounds else None
            ),
            system_prompt_override=self.system_prompt_override,
            aspect_ratio=self.aspect_ratio,
        )


@dataclass
class BenchmarkInput:
    """One input image for a benchmark case."""

    label: str
    role: str  # "user"|"design"|"product"|"area"
    pil_image: Image.Image
    metadata: Optional[dict] = None


@dataclass
class BenchmarkStep:
    """One instrumented step in a benchmark run.

    Emitted twice per step by the runner / instrumented functions: once with
    ``status="running"`` before the model call, then again with
    ``status="completed"`` (or ``"failed"``) after. The router streams these
    to the frontend as they arrive.
    """

    index: int
    kind: str  # "prompt_initial"|"image_gen"|"eval"|"prompt_retry"|"direct_gen"|"verify"
    model: str
    thinking: Optional[str]
    status: str  # "running"|"completed"|"failed"
    started_at: float
    ended_at: Optional[float] = None
    duration_ms: Optional[int] = None
    prompt_used: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    image_count: Optional[int] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Final result of one benchmark case.

    ``output_images`` is a list of PNG byte strings. In dry-run mode the
    runner never writes to disk/storage/DB; the router saves and serves them.
    On failure, ``status="failed"`` and ``error`` is set; ``steps`` still
    contains whatever was captured before the failure.
    """

    process_id: str
    config: BenchmarkConfig
    inputs_summary: list[dict]  # [{label, role, has_metadata}]
    steps: list[BenchmarkStep]
    output_images: list[bytes]  # PNG bytes
    final_prompt: Optional[str]
    retries: int
    total_duration_ms: int
    total_cost_usd: float
    cost_breakdown: list[dict]  # [{model, cost_usd, tokens_in, tokens_out, calls}]
    status: str  # "completed"|"failed"
    error: Optional[str] = None
