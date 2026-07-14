"""Request ID middleware — correlates logs across a single HTTP/SSE request."""
from __future__ import annotations

import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_REQUEST_ID_HEADER = "X-Request-ID"
_request_id_ctx: ContextVar[str | None] = ContextVar("ducon_request_id", default=None)


def get_request_id() -> str | None:
    """Return the request id for the current async context, if any."""
    return _request_id_ctx.get()


def set_request_id(request_id: str | None):
    """Set (or clear) the contextvar; returns a reset token."""
    return _request_id_ctx.set(request_id)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assign ``X-Request-ID`` (honor inbound) and expose via contextvar."""

    async def dispatch(self, request: Request, call_next) -> Response:
        inbound = (request.headers.get(_REQUEST_ID_HEADER) or "").strip()
        request_id = inbound[:64] if inbound else uuid.uuid4().hex
        request.state.request_id = request_id
        token = _request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
        finally:
            _request_id_ctx.reset(token)
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response
