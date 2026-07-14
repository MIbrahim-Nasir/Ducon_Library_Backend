"""E2E configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_BASE_URL = "https://app.duconodl.com"
FALLBACK_BASE_URL = "https://app.ducon.com"


@dataclass(frozen=True)
class E2EConfig:
    base_url: str
    email: str
    password: str
    headless: bool
    allow_gen: bool
    implicit_wait: float
    explicit_wait: float
    page_load_timeout: float


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> E2EConfig:
    return E2EConfig(
        base_url=os.getenv("E2E_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        email=os.getenv("E2E_EMAIL", "test@gmail.com"),
        password=os.getenv("E2E_PASSWORD", "test@123"),
        headless=_truthy("E2E_HEADLESS", "0"),
        allow_gen=_truthy("E2E_ALLOW_GEN", "0"),
        implicit_wait=float(os.getenv("E2E_IMPLICIT_WAIT", "0")),
        explicit_wait=float(os.getenv("E2E_EXPLICIT_WAIT", "25")),
        page_load_timeout=float(os.getenv("E2E_PAGE_LOAD_TIMEOUT", "60")),
    )
