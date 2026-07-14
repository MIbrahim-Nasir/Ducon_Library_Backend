"""Cache flush on landing — backend meta endpoint, middleware, static helper."""
from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_MAIN = ROOT.parent / "Ducon_Library" / "src" / "main.jsx"
FRONTEND_FLUSH = ROOT.parent / "Ducon_Library" / "src" / "utils" / "flushStaleCache.js"
STATIC_FLUSH = ROOT / "app" / "static" / "flushStaleCache.ts"


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.delenv("APP_BUILD_ID", raising=False)
    import app.build_meta as build_meta

    importlib.reload(build_meta)
    from fastapi import FastAPI
    from app.middleware.cache_control import HtmlNoCacheMiddleware
    from app.routers.meta import router as meta_router

    app = FastAPI()
    app.add_middleware(HtmlNoCacheMiddleware)
    app.include_router(meta_router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_meta_build_returns_no_store_headers(client):
    resp = await client.get("/meta/build")
    assert resp.status_code == 200
    data = resp.json()
    assert data["build_id"] == "dev"
    assert data["cache_version"] == data["build_id"]
    cache_control = resp.headers.get("cache-control", "")
    assert "no-store" in cache_control
    assert "no-cache" in cache_control


def test_get_build_id_reads_app_build_id_env(monkeypatch):
    monkeypatch.setenv("APP_BUILD_ID", "deploy-abc123")
    import app.build_meta as build_meta

    importlib.reload(build_meta)
    assert build_meta.get_build_id() == "deploy-abc123"


@pytest.mark.asyncio
async def test_html_responses_get_no_cache_headers():
    from app.middleware.cache_control import HtmlNoCacheMiddleware

    async def call_next(_request: Request):
        return HTMLResponse("<!doctype html><html><body>ok</body></html>")

    middleware = HtmlNoCacheMiddleware(app=AsyncMock())
    response = await middleware.dispatch(Request({"type": "http"}), call_next)

    assert "text/html" in response.headers.get("content-type", "")
    cache_control = response.headers.get("cache-control", "")
    assert "no-store" in cache_control
    assert "no-cache" in cache_control


@pytest.mark.asyncio
async def test_json_responses_are_not_forced_no_cache():
    from app.middleware.cache_control import HtmlNoCacheMiddleware

    async def call_next(_request: Request):
        return JSONResponse({"ok": True})

    middleware = HtmlNoCacheMiddleware(app=AsyncMock())
    response = await middleware.dispatch(Request({"type": "http"}), call_next)

    assert response.headers.get("cache-control") is None


def test_static_flush_helper_documents_landing_flow():
    source = STATIC_FLUSH.read_text(encoding="utf-8")
    assert "flushStaleCache" in source
    assert "/meta/build" in source
    assert "caches.keys" in source
    assert "serviceWorker" in source


def test_frontend_bootstrap_calls_flush_stale_cache():
    if not FRONTEND_MAIN.is_file():
        pytest.skip(f"Frontend repo not found at {FRONTEND_MAIN}")
    if not FRONTEND_FLUSH.is_file():
        pytest.skip(f"Frontend flush helper not found at {FRONTEND_FLUSH}")

    main_source = FRONTEND_MAIN.read_text(encoding="utf-8")
    flush_source = FRONTEND_FLUSH.read_text(encoding="utf-8")

    assert "flushStaleCache" in main_source
    assert "flushStaleCache" in flush_source
    assert "/meta/build" in flush_source
    assert "caches.keys" in flush_source
