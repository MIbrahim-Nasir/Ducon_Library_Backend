"""Request-ID middleware + log_error correlation."""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.error_logger import ErrorLogEntry, get_error_logger, log_error
from app.middleware.request_id import RequestIdMiddleware, get_request_id, set_request_id


@pytest.mark.asyncio
async def test_request_id_middleware_sets_and_echoes_header():
    async def call_next(request: Request):
        assert getattr(request.state, "request_id", None)
        assert get_request_id() == request.state.request_id
        return JSONResponse({"rid": get_request_id()})

    middleware = RequestIdMiddleware(app=None)
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
        }
    )
    response = await middleware.dispatch(request, call_next)
    assert "x-request-id" in {k.lower() for k in response.headers.keys()}
    rid = response.headers.get("x-request-id") or response.headers.get("X-Request-ID")
    assert rid
    assert len(rid) >= 8


@pytest.mark.asyncio
async def test_request_id_middleware_honors_inbound_header():
    async def call_next(request: Request):
        return JSONResponse({"rid": get_request_id()})

    middleware = RequestIdMiddleware(app=None)
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"x-request-id", b"client-corr-99")],
        }
    )
    response = await middleware.dispatch(request, call_next)
    assert response.headers.get("x-request-id") == "client-corr-99"
    body = response.body
    assert b"client-corr-99" in body


@pytest.mark.asyncio
async def test_log_error_picks_up_contextvar_request_id(monkeypatch):
    recorded: list[ErrorLogEntry] = []

    class _Capture:
        def record(self, entry: ErrorLogEntry) -> None:
            recorded.append(entry)

    monkeypatch.setattr("app.error_logger._logger", _Capture())
    token = set_request_id("ctx-rid-abc")
    try:
        await log_error("other", "test.source", "boom")
    finally:
        from app.middleware.request_id import _request_id_ctx

        _request_id_ctx.reset(token)

    assert recorded
    assert recorded[0].request_id == "ctx-rid-abc"


@pytest_asyncio.fixture
async def meta_client():
    from fastapi import FastAPI
    from app.middleware.request_id import RequestIdMiddleware
    from app.routers.meta import router as meta_router

    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    app.include_router(meta_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_meta_build_returns_request_id_header(meta_client):
    resp = await meta_client.get("/meta/build", headers={"X-Request-ID": "meta-rid-1"})
    assert resp.status_code == 200
    assert resp.headers.get("x-request-id") == "meta-rid-1"
