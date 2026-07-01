"""
app/sse.py
──────────
Shared Server-Sent Events response headers.

Disables intermediary buffering (nginx / Cloudflare) and keeps the connection
alive so keepalive comments can flow during long-running generations.
"""

SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Encoding": "identity",
    "X-Accel-Buffering": "no",  # disable nginx / Cloudflare proxy buffering
}
