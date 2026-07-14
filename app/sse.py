"""
app/sse.py
──────────
Shared Server-Sent Events response headers.

Disables intermediary buffering where supported and keeps the connection
alive so keepalive comments can flow during long-running generations.

Note: ``X-Accel-Buffering: no`` is honored by nginx/Cloudflare but **ignored by
Apache**. Production Apache must use ``flushpackets=on`` + disable mod_deflate
for ``text/event-stream`` (see VPS_DEPLOYMENT.md / apache.conf.example).
"""

SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "Content-Encoding": "identity",
    "X-Accel-Buffering": "no",  # nginx / Cloudflare; Apache needs flushpackets=on
}
