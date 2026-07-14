# Ducon Library — VPS Deployment Guide (Apache)

Production deployment for the **FastAPI backend** + **Vite/React SPA** behind
**Apache** (HostGator / similar). This guide is Apache-first — nginx/Caddy are
not used in production.

> Most prod failures (SSE hang → Cloudflare 524, voice disconnects, "generation
> failed but image exists") come from Apache buffering `text/event-stream` or
> missing WebSocket tunnel / long `ProxyTimeout`. Configure §4 carefully.

---

## 1. Architecture

```
                 ┌────────────────────────────────────────────┐
   Browser  ──►  │  Apache + TLS (HTTPS) / Cloudflare orange   │
                 │   • serves SPA static files (dist/)         │
                 │   • proxies API + SSE + WS → :8000          │
                 └───────────────┬────────────────────────────┘
                                 │
                 ┌───────────────▼───────────────┐
                 │  gunicorn + UvicornWorker :8000│
                 │   • REST + SSE generation      │
                 │   • /ws/voice → Gemini Live     │
                 └───┬───────────┬───────────┬────┘
                     │           │           │
              Postgres     Gemini API    Cloudflare R2
```

| Feature | Endpoint(s) | Transport | Apache needs |
|---|---|---|---|
| Voice | `GET /ws/voice` | WebSocket | `mod_proxy_wstunnel`, long timeout |
| Multi-image / autogen / studio / chat / designer events | SSE | `flushpackets=on`, no deflate, `ProxyTimeout 600` |
| REST | `/auth`, `/generations`, … | normal | `ProxyTimeout 120` |

Generation can take **up to ~10 minutes**. Backend emits `: keepalive` every 10 s;
those bytes must reach Cloudflare or the orange cloud kills idle origins (~100 s).

---

## 2. Hard requirements

1. **HTTPS** — mic needs a secure context.
2. **`mod_proxy` + `mod_proxy_http` + `mod_proxy_wstunnel`** enabled.
3. **SSE must not be buffered or gzipped** — disable `mod_deflate` for
   `text/event-stream` (or for the SSE locations entirely). Apache ignores
   nginx-only `X-Accel-Buffering: no`; you must set `flushpackets=on`.
4. **`ProxyTimeout 600`** (or higher) on SSE + WS locations.
5. **`uvicorn[standard]`** + gunicorn `--timeout 0`.
6. **`TRUSTED_PROXY_CIDRS`** — set to Apache / Cloudflare peer ranges so
   `CF-Connecting-IP` is trusted (default is loopback-only).

---

## 3. Backend setup

### 3.1 Packages

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip postgresql \
  apache2 libapache2-mod-proxy-html
sudo a2enmod proxy proxy_http proxy_wstunnel ssl rewrite headers deflate
```

### 3.2 App + venv

```bash
cd /home/appducon/Ducon_Library_Backend   # adjust path
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install "uvicorn[standard]" gunicorn
```

### 3.3 Environment (`.env`)

```bash
ENV=production
GOOGLE_API_KEY=...
JWT_SECRET_KEY=...          # also signs guest cookies if GUEST_SESSION_SECRET unset
DATABASE_URL=postgresql+asyncpg://...
USE_CLOUD_STORAGE=true
# GENERATION_IMAGE_SERVE_MODE=redirect   # or proxy if R2 302s break thumbnails
# R2 private-bucket CORS (browser fetch of presigned URLs / 302 follow). Never use "*".
# R2_CORS_ORIGINS=https://app.duconodl.com,http://localhost:5174,http://127.0.0.1:5174
# Then once per bucket (or after origin changes): python -m scripts.set_r2_cors
# GUEST_SESSION_SECRET=...               # optional; falls back to JWT_SECRET_KEY
GUEST_IP_TOTAL_LIMIT=0
GUEST_SUBNET_LIMIT=0
TRUSTED_PROXY_CIDRS=127.0.0.1/32,::1/128
# If Apache is not on loopback from gunicorn's view, add its peer, e.g.:
# TRUSTED_PROXY_CIDRS=127.0.0.1/32,::1/128,10.0.0.0/8
# Or temporarily: TRUST_FORWARDED_IP_HEADERS=true  (less safe)
APP_BUILD_ID=$(date -u +%Y%m%d%H%M%S)
```

Confirm admin `app_settings` namespace `guest` also has IP/subnet caps at **0**.

### 3.4 systemd

`/etc/systemd/system/ducon-library.service`:

```ini
[Unit]
Description=Ducon FastAPI Uvicorn Service
After=network.target postgresql.service

[Service]
User=appducon
Group=appducon
WorkingDirectory=/home/appducon/Ducon_Library_Backend
EnvironmentFile=/home/appducon/Ducon_Library_Backend/.env
ExecStart=/home/appducon/Ducon_Library_Backend/.venv/bin/gunicorn app.main:app \
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
sudo systemctl enable --now ducon-library
sudo journalctl -u ducon-library -f
```

> Multi-worker note: after `chat_sessions` / `generation_jobs` / `revoked_jtis`
> exist on Postgres (see Local_Files/prodtodo.md §0.A), `-w 3` is fine without
> sticky sessions. Until that migration is applied, prefer sticky sessions or
> `-w 1` so in-memory chat/voice state is not split across workers.

---

## 4. Apache reverse proxy (required)

Example vhost — also saved as `apache.conf.example` in this repo.

```apache
<VirtualHost *:443>
    ServerName app.duconodl.com

    SSLEngine on
    SSLCertificateFile      /etc/letsencrypt/live/app.duconodl.com/fullchain.pem
    SSLCertificateKeyFile   /etc/letsencrypt/live/app.duconodl.com/privkey.pem

    DocumentRoot /var/www/ducon/dist

    # Large uploads (user photos / multi-image)
    LimitRequestBody 62914560

    # ---- SPA static ----
    <Directory /var/www/ducon/dist>
        Options -Indexes +FollowSymLinks
        AllowOverride All
        Require all granted
        # Fallback to index.html for client routes (see .htaccess)
    </Directory>

    # ---- WebSocket voice ----
    ProxyPreserveHost On
    ProxyTimeout 3600
    RewriteEngine On
    RewriteCond %{HTTP:Upgrade} =websocket [NC]
    RewriteRule ^/ws/(.*)$  ws://127.0.0.1:8000/ws/$1 [P,L]
    ProxyPass        /ws/  ws://127.0.0.1:8000/ws/
    ProxyPassReverse /ws/  ws://127.0.0.1:8000/ws/

    # ProxyTimeout must be at VirtualHost level on HostGator EA4 (not inside LocationMatch).
    ProxyTimeout 600

    # ---- SSE (no buffering, long timeout, no gzip) ----
    # CRITICAL: without flushpackets=on, Apache buffers the stream, Cloudflare
    # sees an idle origin, and kills the connection (~100s) while the server
    # keeps generating (quota burned, client sees error).
    <LocationMatch "^/(generate-multi-image|chat|studio/directions|designer/jobs/.+/events|generation/jobs/.+/events)">
        ProxyPass        http://127.0.0.1:8000 flushpackets=on
        ProxyPassReverse http://127.0.0.1:8000
        SetEnv no-gzip 1
        SetEnv no-brotli 1
        Header set Cache-Control "no-cache"
        Header set X-Accel-Buffering "no"
    </LocationMatch>

    # Disable mod_deflate for event-stream (belt and braces)
    <IfModule mod_deflate.c>
        SetEnvIfNoCase Content-Type ^text/event-stream$ no-gzip dont-vary
    </IfModule>

    # ---- Normal API ----
    ProxyPass        /auth        http://127.0.0.1:8000/auth
    ProxyPassReverse /auth        http://127.0.0.1:8000/auth
    ProxyPass        /bookmarks   http://127.0.0.1:8000/bookmarks
    ProxyPassReverse /bookmarks   http://127.0.0.1:8000/bookmarks
    ProxyPass        /generations http://127.0.0.1:8000/generations
    ProxyPassReverse /generations http://127.0.0.1:8000/generations
    ProxyPass        /guest       http://127.0.0.1:8000/guest
    ProxyPassReverse /guest       http://127.0.0.1:8000/guest
    ProxyPass        /images      http://127.0.0.1:8000/images
    ProxyPassReverse /images      http://127.0.0.1:8000/images
    ProxyPass        /public      http://127.0.0.1:8000/public
    ProxyPassReverse /public      http://127.0.0.1:8000/public
    ProxyPass        /quotation   http://127.0.0.1:8000/quotation
    ProxyPassReverse /quotation   http://127.0.0.1:8000/quotation
    ProxyPass        /search      http://127.0.0.1:8000/search
    ProxyPassReverse /search      http://127.0.0.1:8000/search
    ProxyPass        /contact     http://127.0.0.1:8000/contact
    ProxyPassReverse /contact     http://127.0.0.1:8000/contact
    ProxyPass        /meta        http://127.0.0.1:8000/meta
    ProxyPassReverse /meta        http://127.0.0.1:8000/meta
    # Admin API only — bare /admin is the React SPA
    ProxyPass        /admin/      http://127.0.0.1:8000/admin/
    ProxyPassReverse /admin/      http://127.0.0.1:8000/admin/
    ProxyPass        /designer    http://127.0.0.1:8000/designer
    ProxyPassReverse /designer    http://127.0.0.1:8000/designer

    # Default ProxyTimeout for REST
    <LocationMatch "^/(auth|bookmarks|generations|guest|images|public|quotation|search|contact|meta|admin/)">
        ProxyTimeout 120
    </LocationMatch>
</VirtualHost>

# HTTP → HTTPS
<VirtualHost *:80>
    ServerName app.duconodl.com
    Redirect permanent / https://app.duconodl.com/
</VirtualHost>
```

SPA fallback `.htaccess` in `dist/`:

```apache
RewriteEngine On
RewriteBase /
RewriteRule ^index\.html$ - [L]
RewriteCond %{REQUEST_FILENAME} !-f
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule . /index.html [L]
```

```bash
sudo apachectl configtest && sudo systemctl reload apache2
```

---

## 5. Frontend deploy

```bash
cd /path/to/Ducon_Library
export VITE_API_BASE_URL=same-origin
export VITE_APP_BUILD_ID=$(date -u +%Y%m%d%H%M%S)
npm ci && npm run build
sudo rsync -a --delete dist/ /var/www/ducon/dist/
# Keep APP_BUILD_ID on the backend in sync with VITE_APP_BUILD_ID
```

Guest identity: the SPA calls `POST /guest/session` (HttpOnly cookie) and sends
`X-Guest-Fingerprint`. Ensure `credentials: 'include'` reaches Apache same-origin.

---

## 6. Cloudflare (orange cloud)

- Bypass cache for: `/ws/*`, `/generate-multi-image*`,
  `/chat*`, `/studio/directions*`, `/designer/jobs*`, `/auth*`, `/generations*`,
  `/guest*`, `/meta*`
- WebSockets: enabled by default
- Idle timeout ~100 s — Apache must flush SSE keepalives (§4)

---

## 7. Smoke tests

```bash
# REST
curl -sS https://app.duconodl.com/meta/build

# SSE must trickle (keepalive comments), NOT dump at the end:
curl -N -H "Authorization: Bearer <token>" \
     -F prompt="test" -F images_meta='[{"label":"x","source":"1"}]' \
     https://app.duconodl.com/generate-multi-image

# WebSocket upgrade → 101
curl -i -N \
  -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: dGhlIHNhbXBsZQ==" \
  "https://app.duconodl.com/ws/voice?token=<jwt>"
```

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Generation errors ~100 s; image appears later in list | Apache buffered SSE; CF idle kill | `flushpackets=on`, disable gzip on SSE, `ProxyTimeout 600` |
| Voice stuck "connecting" | No WS tunnel | `mod_proxy_wstunnel` + Upgrade rewrite |
| Voice drops every ~10 min | Old client tore down on `go_away` | Deploy frontend that ignores `go_away` (backend self-heals) |
| Office shares one guest limit | `GUEST_IP_TOTAL_LIMIT` > 0 or no cookie/fingerprint | Set caps to 0; deploy FE guest session + fingerprint |
| `GET /generations/{id}/image` → 302 | Normal with R2 | FE must fetch with `credentials: 'omit'` (Bearer still sent). If R2 CORS is wrong: `R2_CORS_ORIGINS=https://app.duconodl.com,... python -m scripts.set_r2_cors`. Last resort: `GENERATION_IMAGE_SERVE_MODE=proxy` |
| CORS: ACAO `*` vs credentials include | Blob fetch followed 302 to R2 with cookies | Redeploy FE with omit-credentials blob helpers; re-apply R2 CORS (never `*`) |
| `Error while closing socket [Errno 9]` on restart | Harmless gunicorn/uvicorn race | Ignore if shutdown completes |
| Chat context lost across requests | Multi-worker memory | Sticky sessions or `-w 1` until Redis sessions |

---

## 9. Deploy checklist

- [ ] `ENV=production`, secrets set, `GUEST_IP_TOTAL_LIMIT=0`, `GUEST_SUBNET_LIMIT=0`
- [ ] DB migration: `source_image_url` on `generations` / `guest_generations` (see `Local_Files/prodtodo.md`)
- [ ] Apache SSE + WS config live; `curl -N` shows trickle keepalives
- [ ] systemd `--timeout 0 --graceful-timeout 30`
- [ ] SPA built with `VITE_API_BASE_URL=same-origin` + matching `VITE_APP_BUILD_ID` / `APP_BUILD_ID`
- [ ] Cloudflare cache bypass for dynamic/SSE/WS paths
- [ ] `TRUSTED_PROXY_CIDRS` includes Apache peer
- [ ] Smoke tests in §7 pass
