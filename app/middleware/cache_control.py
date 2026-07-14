"""HTTP cache-control helpers for HTML shell responses."""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_HTML_NO_CACHE = "no-cache, no-store, must-revalidate"


class HtmlNoCacheMiddleware(BaseHTTPMiddleware):
    """Prevent browsers and reverse proxies from caching HTML entrypoints."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type.lower():
            response.headers["Cache-Control"] = _HTML_NO_CACHE
            response.headers["Pragma"] = "no-cache"
        return response
