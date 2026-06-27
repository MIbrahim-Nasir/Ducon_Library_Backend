# Ducon Library — VPS Deployment Guide

Production deployment notes for the **FastAPI backend** + **Vite/React SPA**, with
special attention to the two features that break under a naïve reverse-proxy
setup: the **voice agent (WebSocket)** and **image generation (long-running SSE)**.

> The Vite dev-server proxy (`vite.config.js`) only applies in development.
> **In production there is no Vite proxy** — your reverse proxy (nginx/Caddy)
> must route everything. Most "works locally, fails on the VPS / other devices"
> bugs come from missing WebSocket upgrade or SSE buffering rules below.

---

## 1. Architecture

```
                 ┌────────────────────────────────────────────┐
   Browser  ──►  │  Reverse proxy (nginx/Caddy) + TLS (HTTPS)  │
                 │   • serves SPA static files (dist/)         │
                 │   • proxies API paths → FastAPI :8000       │
                 │   • proxies /ws/voice  → FastAPI (WebSocket)│
                 └───────────────┬────────────────────────────┘
                                 │
                 ┌───────────────▼───────────────┐
                 │  FastAPI (uvicorn) :8000       │
                 │   • REST + SSE generation      │
                 │   • /ws/voice → Gemini Live     │
                 │   • Chroma (vector search)     │
                 └───┬───────────┬───────────┬────┘
                     │           │           │
              Postgres     Gemini API    Cloudflare R2 (optional)
```

Key runtime traits that drive the proxy config:

| Feature | Endpoint(s) | Transport | Needs |
|---|---|---|---|
| Voice agent | `GET /ws/voice?token=…` | **WebSocket** | upgrade headers, long read timeout |
| Multi-image gen | `POST /generate-multi-image` | **SSE** (keepalive every 10 s) | no buffering, long timeout |
| Single gen / Studio | `POST /autogenerate-images` | **SSE** | no buffering, long timeout |
| Chat | `POST /chat/*` | **SSE** | no buffering |
| Catalog images | `GET /public/images/*`, `/images/*` | static/REST | normal |
| Everything else | `/auth /bookmarks /generations /guest /quotation /designer /studio /search` | REST | normal |

Generation can take **up to ~10 minutes** (`GENERATION_TIMEOUT_MS = 600000` on the
client). The backend emits `: keepalive` SSE comments every 10 s so the connection
never idles out, but the proxy still needs a long read timeout (see §4).

---

## 2. Hard requirements / gotchas

1. **HTTPS is mandatory.** The microphone (`getUserMedia`) only works in a *secure
   context*. Over plain `http://` (except `localhost`), the browser blocks the mic
   and the voice agent will never start on other devices. Terminate TLS at the
   proxy (Let's Encrypt / Cloudflare).
2. **WebSocket upgrade must be proxied for `/ws/`.** Without it, the voice agent
   sits in "connecting" forever.
3. **SSE locations must disable proxy buffering** (`proxy_buffering off` /
   `X-Accel-Buffering: no`) and **must not be gzipped**, or the browser receives
   nothing until the whole response ends (looks like a hang/timeout). The backend
   already sends `Content-Encoding: identity` + `X-Accel-Buffering: no`.
4. **Long read/idle timeouts** (≥ 600 s) on generation + WS locations.
5. **Run uvicorn with WebSocket support** — install `uvicorn[standard]` (pulls in
   `websockets` + `httptools`).

---

## 3. Backend setup

### 3.1 System packages

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip postgresql nginx
```

### 3.2 App + dependencies

```bash
cd /opt
sudo git clone <your-repo> ducon-backend
cd ducon-backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install "uvicorn[standard]" gunicorn   # ws support + process manager
```

### 3.3 Database (Postgres)

```bash
sudo -u postgres psql -c "CREATE USER ducon WITH PASSWORD 'strong-password';"
sudo -u postgres psql -c "CREATE DATABASE \"Ducon_Library\" OWNER ducon;"
```

Set `DATABASE_URL=postgresql+asyncpg://ducon:strong-password@localhost/Ducon_Library`.
Ensure your table-creation/migration step has run before first traffic.

### 3.4 Writable data directories

The backend writes generated images to `outputs/`, persists the vector DB to
`chroma_db/`, and reads the catalog from `data/images/` (incl. `metadata.json`).
Make sure these exist and are writable by the service user:

```bash
mkdir -p outputs chroma_db data/images
sudo chown -R www-data:www-data /opt/ducon-backend
```

### 3.5 Environment file (`/opt/ducon-backend/.env`)

```ini
# ─── Core / AI ───────────────────────────────────────────────
GOOGLE_API_KEY=your_gemini_api_key
JWT_SECRET_KEY=generate-a-long-random-secret        # REQUIRED in prod
GOOGLE_CLIENT_ID=your_google_oauth_client_id        # if using Google login

# ─── Database ────────────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://ducon:strong-password@localhost/Ducon_Library

# ─── Models (override only if needed) ────────────────────────
IMAGE_GEN_MODEL=gemini-3-pro-image-preview
MULTI_IMAGE_PRO_MODEL=gemini-3-pro-image-preview
MULTI_IMAGE_FLASH_MODEL=gemini-3.1-flash-image-preview
CHAT_MODEL=gemini-3.5-flash
LIVE_MODEL=gemini-3.1-flash-live-preview
LIVE_VOICE=Kore
QUOTATION_MODEL=gemini-3.1-pro-preview

# ─── Image-gen QC pipeline ───────────────────────────────────
GEN_EVAL_MAX_ROUNDS=3
PROMPT_VERIFY_MAX_ROUNDS=2
# Keep OFF in prod — auto-learn can rewrite/degrade the system prompt.
IMAGE_GEN_AGENT_AUTO_LEARN=false

# ─── Claude (Anthropic) provider — optional ──────────────────
# USE_CLAUDE=true routes the TEXT-reasoning agents (chat agent, image-gen prompt
# writer + verifier + QC evaluator, designer job agent) through Claude instead of
# Gemini. Gemini is ALWAYS kept for AI search/embeddings, the voice agent (Live),
# the Studio directions agent, and the image-generation models (Nano Banana).
# Requires the `anthropic` package (already in requirements.txt) + an API key.
USE_CLAUDE=false
ANTHROPIC_API_KEY=your_anthropic_api_key   # required only when USE_CLAUDE=true
CLAUDE_MODEL=claude-sonnet-4-6
CLAUDE_THINKING=adaptive                    # adaptive(high) | off | <int budget_tokens>
CLAUDE_MAX_TOKENS=16000
CLAUDE_CHAT_MAX_MESSAGES=60

# ─── Guest limits / bot protection ───────────────────────────
GUEST_GEN_LIMIT=3
GUEST_IP_GEN_LIMIT=6
TURNSTILE_SECRET_KEY=your_turnstile_secret          # if guests can generate

# ─── Cloudflare R2 (optional — else local disk under outputs/) ─
USE_CLOUD_STORAGE=false
# R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
# R2_ACCESS_KEY_ID=...
# R2_SECRET_ACCESS_KEY=...
# R2_PRIVATE_BUCKET=ducon-private
# PRESIGNED_URL_EXPIRY=3600

# ─── Misc ────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB=50
# LIVE_DEBUG=false
```

> All env vars are optional except `GOOGLE_API_KEY`, `JWT_SECRET_KEY`, and
> `DATABASE_URL`. Sensible defaults exist for the rest (see source). Other tunables:
> `CHAT_THINKING_LEVEL`, `LIVE_THINKING_LEVEL`, `DESIGNER_AGENT_*`,
> `STUDIO_DIRECTIONS_*`, `MULTI_IMAGE_THINKING_LEVEL`.

### 3.6 CORS

`app/main.py` currently sets `allow_origins=["*"]`. If you deploy the SPA and API
on the **same domain** (recommended, §5), CORS is irrelevant. If you split them
(API on `api.yourdomain.com`), tighten it:

```python
allow_origins=["https://yourdomain.com"],
allow_credentials=True,
```

(`allow_origins=["*"]` + `allow_credentials=True` is rejected by browsers for
credentialed requests.)

### 3.7 Run as a service (systemd)

`/etc/systemd/system/ducon-backend.service`:

```ini
[Unit]
Description=Ducon FastAPI backend
After=network.target postgresql.service

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/ducon-backend
EnvironmentFile=/opt/ducon-backend/.env
# UvicornWorker = async; -w 2..4 workers. WS connections are per-worker (no shared
# state needed). --timeout 0 so long async generations are never worker-killed.
ExecStart=/opt/ducon-backend/.venv/bin/gunicorn app.main:app \
    -k uvicorn.workers.UvicornWorker \
    -w 3 -b 127.0.0.1:8000 \
    --timeout 0 --graceful-timeout 30 --keep-alive 75
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ducon-backend
sudo journalctl -u ducon-backend -f      # watch logs
```

---

## 4. Reverse proxy — nginx (recommended, single domain)

This serves the SPA and proxies all API/WS/SSE paths on **one origin**
(`yourdomain.com`), which avoids CORS entirely. Set the frontend
`VITE_API_BASE_URL=same-origin` (see §6).

`/etc/nginx/sites-available/ducon`:

```nginx
# Upstream FastAPI
upstream ducon_api { server 127.0.0.1:8000; }

# WebSocket upgrade map
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Allow large uploads (user photos / multi-image)
    client_max_body_size 60m;

    # ---- SPA static files ----
    root /var/www/ducon/dist;
    index index.html;

    # ---- 1. Voice agent WebSocket ----
    location /ws/ {
        proxy_pass http://ducon_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host       $host;
        proxy_set_header X-Real-IP  $remote_addr;
        proxy_read_timeout  3600s;   # keep idle WS alive
        proxy_send_timeout  3600s;
    }

    # ---- 2. Long-running SSE generation + chat ----
    location ~ ^/(generate-multi-image|autogenerate-images|chat) {
        proxy_pass http://ducon_api;
        proxy_http_version 1.1;
        proxy_set_header Host       $host;
        proxy_set_header X-Real-IP  $remote_addr;
        proxy_set_header Connection "";       # keepalive to upstream
        proxy_buffering off;                  # CRITICAL for SSE
        proxy_cache off;
        proxy_read_timeout  600s;             # >= client 10-min budget
        proxy_send_timeout  600s;
        chunked_transfer_encoding on;
    }

    # ---- 3. Normal REST API ----
    location ~ ^/(auth|bookmarks|generations|guest|images|public|quotation|designer|studio|search) {
        proxy_pass http://ducon_api;
        proxy_http_version 1.1;
        proxy_set_header Host      $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }

    # ---- 4. SPA fallback (client-side routing) ----
    location / {
        try_files $uri $uri/ /index.html;
    }
}

# HTTP → HTTPS redirect
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$host$request_uri;
}
```

```bash
sudo ln -s /etc/nginx/sites-available/ducon /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
# TLS:
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

> **Why `proxy_buffering off` on the SSE block:** with buffering on, nginx holds
> the whole streamed body and the browser sees nothing until the end — the
> generation looks frozen/timed-out. The backend also emits `X-Accel-Buffering: no`
> which nginx honours, but setting it explicitly is belt-and-braces.

---

## 5. Reverse proxy — Caddy (simplest alternative)

Caddy auto-provisions TLS and proxies WebSockets transparently — no upgrade-header
boilerplate. `/etc/caddy/Caddyfile`:

```caddy
yourdomain.com {
    encode gzip

    # SSE + WS must not be buffered/compressed
    @stream path /ws/* /generate-multi-image* /autogenerate-images* /chat*
    reverse_proxy @stream 127.0.0.1:8000 {
        flush_interval -1          # disable buffering (stream immediately)
        transport http {
            read_timeout  600s
            write_timeout 600s
        }
    }

    # Other API paths
    @api path /auth* /bookmarks* /generations* /guest* /images* /public* \
              /quotation* /designer* /studio* /search*
    reverse_proxy @api 127.0.0.1:8000

    # SPA static + fallback
    root * /var/www/ducon/dist
    try_files {path} /index.html
    file_server
}
```

---

## 6. Frontend (SPA) build + deploy

```bash
cd /path/to/Ducon_Library          # the frontend repo
# .env (production):
#   VITE_API_BASE_URL=same-origin        # same domain as the SPA (recommended)
#   VITE_GOOGLE_CLIENT_ID=...            # if using Google login
#   VITE_TURNSTILE_SITE_KEY=...          # if guests generate
npm ci
npm run build                      # → dist/
sudo rsync -a --delete dist/ /var/www/ducon/dist/
```

- `VITE_API_BASE_URL=same-origin` makes the client call the **current origin** for
  all API calls, and derive the WS URL as `wss://<origin>/ws/voice` — which is why
  the nginx/Caddy `/ws/` rule must exist.
- If you split onto an `api.` subdomain instead, set
  `VITE_API_BASE_URL=https://api.yourdomain.com`, give that host its own server
  block proxying **everything** (`location / { … }` incl. `/ws/`) to `:8000`, and
  tighten CORS (§3.6).

---

## 7. Cloudflare (if proxying through the orange cloud)

- **WebSockets:** enabled by default on all plans — no action needed. The 10 s SSE
  keepalives in the backend keep generation streams under Cloudflare's ~100 s idle
  limit, so no extra timeout config is required.
- **Caching:** add a Cache Rule to **Bypass cache** for
  `/ws/*`, `/generate-multi-image*`, `/autogenerate-images*`, `/chat*`,
  `/auth*`, `/generations*` (anything dynamic / streaming / authed).
- **Cloudflare Tunnel** (no public IP) works too — `cloudflared` forwards WS and
  SSE transparently; point the tunnel at your nginx/Caddy (or directly at `:8000`
  if you let cloudflared serve TLS).

---

## 8. Smoke tests

```bash
# Backend up
curl -sS https://yourdomain.com/images/metadata | head -c 200

# SSE streams incrementally (you should see `data:`/`: keepalive` lines trickle in,
# NOT a single dump at the end):
curl -N -H "Authorization: Bearer <token>" \
     -F prompt="test" -F images_meta='[{"label":"x","source":"1"}]' \
     https://yourdomain.com/generate-multi-image

# WebSocket upgrade returns 101 (needs a recent curl with --include):
curl -i -N \
  -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: dGhlIHNhbXBsZQ==" \
  "https://yourdomain.com/ws/voice?token=<jwt>"
```

In the browser, confirm the page is `https://`, then open the voice agent — it
should move from **connecting → listening** within a couple of seconds.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Voice stuck in **"connecting"** | `/ws/` not proxied / no WS upgrade | Add the `/ws/` location (§4) or use Caddy; check 101 in `curl -i` |
| Voice never starts, no mic prompt | Not served over HTTPS | Terminate TLS; mic needs a secure context |
| Generation **hangs then errors** | proxy buffering SSE / gzip on stream | `proxy_buffering off` + no gzip on SSE locations; long `proxy_read_timeout` |
| Generation **502/504** after ~30–100 s | proxy/Cloudflare idle timeout | keepalives are in code; raise `proxy_read_timeout`, bypass CF cache |
| `413 Request Entity Too Large` on upload | body limit | `client_max_body_size 60m;` |
| CORS errors (split domains) | wildcard + credentials | set explicit `allow_origins` (§3.6) |
| Worker killed mid-generation | gunicorn `--timeout` | use `--timeout 0` for UvicornWorker (§3.7) |

---

## 10. Deploy / update checklist

- [ ] `GOOGLE_API_KEY`, `JWT_SECRET_KEY`, `DATABASE_URL` set in backend `.env`
- [ ] `IMAGE_GEN_AGENT_AUTO_LEARN=false`
- [ ] If `USE_CLAUDE=true`: `ANTHROPIC_API_KEY` set and `anthropic` installed (`pip install -r requirements.txt`)
- [ ] `outputs/`, `chroma_db/`, `data/images/` exist and are writable
- [ ] `uvicorn[standard]` installed (WebSocket support)
- [ ] systemd service running with `--timeout 0`
- [ ] nginx/Caddy: `/ws/` WS rule + SSE no-buffering rule + long timeouts
- [ ] TLS (HTTPS) active — required for mic
- [ ] SPA built with `VITE_API_BASE_URL=same-origin` and deployed to `dist/`
- [ ] Cloudflare: cache bypass for dynamic/streaming/WS paths (if used)
- [ ] Smoke tests in §8 pass (REST + SSE trickle + WS 101 + voice connects)
```
