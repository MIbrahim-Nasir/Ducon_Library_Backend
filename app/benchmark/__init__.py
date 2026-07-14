"""Internal AI-generation benchmark pipeline + runner.

This package owns:
- ``types``: dataclasses exchanged with the dev router and frontend.
- ``runner``: the ``run_benchmark_case`` coroutine the dev router calls.

The dev router (``app/routers/dev_benchmark.py``) and the result store
(``app/benchmark/store.py``) are owned by other workers.
"""
from app.benchmark.types import (
    BenchmarkConfig,
    BenchmarkInput,
    BenchmarkResult,
    BenchmarkStep,
    RunOverrides,
    StepMetrics,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkInput",
    "BenchmarkResult",
    "BenchmarkStep",
    "RunOverrides",
    "StepMetrics",
]
