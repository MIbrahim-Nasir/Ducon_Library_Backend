# Changelog — Ducon Library Backend

**Session date:** 2026-07-11 (updated) / 2026-07-10 / 2026-07-09 / 2026-07-08  
**Branch:** `FastprodCloudflare` (uncommitted working tree at time of writing)  
**Scope:** Watermarks, AI search image input, generation pipeline audit, guest identity & limits, abuse protection, cache flush, cookies, platform hardening, tests, Resend burst handling, Selenium E2E.

### 2026-07-11 — Selenium E2E suite (production UI smoke)

- **`tests/e2e/`** — Selenium + pytest page-object suite against `https://app.duconodl.com` (fallback `app.ducon.com` if primary fails).
- Covers guest consent/land, catalog tabs, bookmarks, uploads (fixture PNG), AI Generations panel, chat UI, studio wizard steps (no paid visualize), login/logout.
- Markers: `@pytest.mark.e2e` (excluded from default `pytest tests/`); gen paths gated by `E2E_ALLOW_GEN=1`.
- Env: `E2E_BASE_URL`, `E2E_EMAIL`, `E2E_PASSWORD`, `E2E_HEADLESS`, `E2E_ALLOW_GEN`.
- Deps: `requirements-e2e.txt`. Run book: `tests/e2e/README.md`.
- Turnstile: detect + skip with clear message when Cloudflare blocks automation.

### 2026-07-10 — Designer final-summary markdown hint

- **Prompt** (`app/prompts/designer-final-summary.md`) — allow light markdown in user-facing final summaries (frontend now renders it in the designer job card).

### 2026-07-10 — Corner watermark size & opacity

- **Corner watermark** (`app/watermark.py`) — always composites at **100% opacity** (ignores `WATERMARK_OPACITY`, which still only affects the centered “D”); scale bumped ~20% (`_CORNER_WIDTH_RATIO` 0.22→0.26, `_CORNER_MAX_HEIGHT_RATIO` 0.12→0.14). Admin catalog + `tests/test_watermark.py` updated.

### 2026-07-10 — Production generation failures (verify + FE refresh)

- **Backend (already in tree; verified)** — guest fingerprint remapping uses canonical `session_id` in `multi_image_gen.py`, `main.py`, and `chat.py`; guest `gen:N` lookups scoped to `GuestGeneration`; empty Gemini candidates raise a clear `RuntimeError` instead of a cryptic failure.
- **Backend (small follow-up)** — `_compact_generations_list` accepts `{ generations: [...] }` payloads (guest note) as well as raw arrays.
- **Frontend (Ducon_Library)** — after chat `generate_multi_image`, dispatch `ducon:generations-updated` so the AI Generations modal refreshes; `open_ai_generations` no longer swallows list errors as `[]` (throws + surfaces Retry in the modal).

### 2026-07-10 — Watermarks, AI search image input, generation pipeline

- **Watermarks** (`app/watermark.py`, `app/static/ducon_*.png`)
  - Replaced diagonal/adaptive logo + old `corner watermark.jpeg` with new assets:
    - `ducon_main_watermark.png` — large centered Ducon “D”, scaled to fit (no cut-off)
    - `ducon_corner_watermark.png` — flush bottom-left (edges align; no padding)
  - Near-black mattes keyed to transparent when assets lack real alpha
  - Admin settings: `WATERMARK_MAIN_PATH` / `WATERMARK_CORNER_PATH` (removed obsolete logo-path knobs)
  - Tests: `tests/test_watermark.py`
- **AISearch image input** (`app/search_tools.py`, chat/voice prompts)
  - Tool schemas accept optional `image_ref` alongside `query` (text, image, or both)
  - Descriptions document that text-only / image-only / multimodal rankings differ
  - `POST /search` already supports `query` + `file` multimodal embeddings — frontend must pass `file` when resolving `image_ref`
  - Tests: `tests/test_ai_search_image_input.py`
- **Generation pipeline (chat/voice → ImageGenAgent)**
  - `/generate-multi-image` explicitly passes `enable_verify=True` (Studio-equivalent prompt+eval+retry)
  - Broader label heuristics so voice shorthand (`user terrace`, `Ducon pergola`) still classifies roles
  - Voice `generate_multi_image` tool schema aligned with Studio label conventions
  - Compact `generate_multi_image` tool_results in chat + live (drop long signed URLs)
  - Prompt guidance updated: quick-gen still uses full ImageGenAgent loop when ≥2 labeled images
  - Tests: ImageGenAgent path + voice label classification in `tests/test_tool_generate_image_regressions.py`

### 2026-07-09 — Resend burst handling & auth rate-limit polish

- **`app/email_service.py`** — throttle Resend to ~6 req/s (configurable `RESEND_MAX_RPS`) with concurrency cap; retry HTTP 429 with backoff / `Retry-After`; document that daily 100/day quota cannot be queued away.
- **`app/otp_service.py`** — on Resend failure after retries, delete the unused OTP so per-email cooldown does not block an immediate retry; other users unaffected.
- **`app/routers/auth.py`** — raise `signup_otp` / `password_forgot` IP limits from 5→10 per 300s (keep IP anti-abuse + per-email OTP cooldown).
- **`app/rate_limiter.py`** — resolve client IP via Cloudflare `CF-Connecting-IP` (same helper as guest identity) so CDN peer IP is not the rate-limit key.
- **Tests** — `tests/test_email_service.py` (throttle/retry); stronger same-IP multi-guest isolation + CF IP rate-limit coverage.

> This document covers **all modified and new files** in the current working tree (`git diff` + untracked additions). Prior commits on the branch (`d85bf8a` … `d43a850`) are not replayed here; this changelog reflects the **delta since last commit**.

---

## Summary

| Metric | Value |
|--------|-------|
| Modified tracked files | 41 (+2,578 / −613 lines) |
| New modules / dirs | `guest_identity`, `guest_session_token`, `build_meta`, `middleware`, `error_logger`, `designer_cleanup`, `benchmark/`, `routers/meta`, `routers/dev_benchmark`, `static/*.ts`, `tests/`, `prodtodo.md`, `scripts/run_deploy_checks.py` |
| New automated tests | **124** test functions across **14** files (new `tests/` suite) |
| New public API endpoints | `POST /guest/session`, `GET /meta/build` |
| DB tables added | `designer_jobs`, `app_error_logs` |
| DB columns added (`guest_sessions`) | `fingerprint_hash`, `composite_hash`, `subnet_key`, `ja4_fingerprint`, `asn` |

---

## 1. Guest identity & per-device limits

### Fingerprint-first quota (UUID rotation resistance)

- New **`app/guest_identity.py`** — resolves client IP (Cloudflare `CF-Connecting-IP` → `True-Client-IP` → `X-Forwarded-For` → peer), subnet key (IPv4 `/24`, IPv6 `/64`), Cloudflare JA4/JA3/ASN/country signals, and builds a server-side `composite_hash` for audit.
- New **`app/guest_usage.py`** overhaul — identity resolution order:
  1. **`fingerprint_hash`** (from `X-Guest-Fingerprint`) — if a row already has this hash, that session is returned **regardless of UUID** in header/cookie.
  2. **`session_id`** — `X-Guest-Session-Id` header or signed cookie when no fingerprint.
- **`composite_hash` / `subnet_key` / `ja4` / `asn` are stored for audit only** — they do **not** merge different UUIDs into one quota row.
- Per-feature limits (configurable via admin settings / env):
  - `GUEST_GEN_LIMIT` = 3 (one saved output image per pipeline; eval/retry rounds do not count)
  - `GUEST_CHAT_LIMIT` = 10 (one completed `/chat/message` SSE turn; `tool_result` continuations do not count)
  - `GUEST_VOICE_LIMIT` = 5 (one completed voice turn)
- Optional shared caps (default **disabled** for office/corporate NAT):
  - `GUEST_IP_TOTAL_LIMIT` = 0
  - `GUEST_SUBNET_LIMIT` = 0
- **`increment_guest_usage`** uses DB-side `col = col + 1` to avoid lost-update races under concurrency.
- **`usage_snapshot`** returns `{ generations, chat, voice }` plus legacy `{ used, limit }` for generation-only clients.

### Frontend fingerprint helper

- New reference script **`app/static/guestFingerprint.ts`** (copy into Ducon_Library frontend):
  - Uses `@fingerprintjs/fingerprintjs` OSS
  - SHA-256 hashes `visitorId` client-side
  - Sends **64-char hex** on `X-Guest-Fingerprint` (never raw visitorId)
  - Survives `localStorage.clear()` / incognito on same device in most browsers

### ORM & schema

- **`GuestSession`** model extended with `fingerprint_hash`, `composite_hash`, `subnet_key`, `ja4_fingerprint`, `asn` (+ indexes).
- **`database_schema.sql`** §5 post-migration ALTERs add the five columns and three indexes idempotently.
- Migration runbook: **[prodtodo.md](prodtodo.md)**.

---

## 2. Guest session cookies (signed HttpOnly)

### New module: `app/guest_session_token.py`

- Cookie name: **`ducon_guest_session`**
- Value format: `{uuid}.{hmac_sha256}` signed with `GUEST_SESSION_SECRET` (falls back to `JWT_SECRET_KEY`)
- Resolution precedence: **`X-Guest-Session-Id` header overrides cookie** when both present
- Cookie attributes: `HttpOnly`, `Secure` in production, `SameSite=none` (prod) / `lax` (dev), 1-year `max_age`, `path=/`
- WebSocket helper: `resolve_guest_session_id_from_parts(headers, cookies)` for voice WS

### New endpoint

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/guest/session` | None | Issues server UUID + sets signed HttpOnly cookie. Returns `{ "session_id": "<uuid>" }`. |

### Updated guest endpoints

| Method | Path | Changes |
|--------|------|---------|
| `GET` | `/guest/usage` | Accepts cookie or header; fingerprint-aware `get_guest_session_row` |
| `GET` | `/guest/gen-count` | Legacy generation-only snapshot; same identity resolution |
| `POST` | `/auth/claim-guest-generations` | Rate-limited (`claim_guest` prefix + user id suffix) |
| `POST` | `/guest/cleanup` | Now also prunes `designer_jobs`; requires `X-Cron-Secret` when `GUEST_CLEANUP_SECRET` set (mandatory in prod) |
| `GET` | `/guest/generations/{id}/image` | Signed-URL gate unchanged |

All guest AI entry points now call `build_guest_request_identity()` and pass identity into `enforce_guest_limit` / `get_or_create_guest_session`.

---

## 3. Abuse protection & bot defense

### Cloudflare Turnstile (fail-closed in production)

- **`verify_turnstile`** in `app/routers/guest.py`:
  - Skipped in development unless `TURNSTILE_ENFORCE=true`
  - **Production without `TURNSTILE_SECRET_KEY` → verification fails** (fail-closed)
  - Required on guest: `/autogenerate-images`, `/chat/message`, `/generate-multi-image`, voice WS (`turnstile_token` query param)
- New **`ENFORCE_TURNSTILE`** in `app/config.py` mirrors frontend `VITE_TURNSTILE_ENFORCE`
- Cloudflare test keys documented in `env_template.txt`

### Rate limiting improvements (`app/rate_limiter.py`)

- `require_rate_limit(..., key_suffix=...)` — isolates per-user limits on shared NAT (e.g. `claim_guest:{ip}:{user_id}`)
- `/search` — 30 req/min per IP
- `/studio/directions` — 15 req/min per IP
- Image upload size guard: 15 MB on search/studio paths

### Structured error logging (`app/error_logger.py`)

- Non-blocking queue → batch insert into **`app_error_logs`**
- Categories: `generation`, `chat`, `voice`, `guest`, `admin`, etc.
- Guest limit / Turnstile failures logged with `guest_session_id`, endpoint, HTTP status
- Started/stopped in app lifespan alongside usage recorder

---

## 4. Cache flush & deploy build metadata

### Backend build id

- New **`app/build_meta.py`** — reads `APP_BUILD_ID` env (falls back to `"dev"`)
- New **`GET /meta/build`** (`app/routers/meta.py`):
  - Response: `{ "build_id": "...", "cache_version": "..." }`
  - Headers: `Cache-Control: no-store, no-cache, must-revalidate`, `Pragma: no-cache`

### HTML shell no-cache middleware

- New **`app/middleware/cache_control.py`** — `HtmlNoCacheMiddleware` sets no-cache on `text/html` responses

### Frontend reference script

- **`app/static/flushStaleCache.ts`** — on landing:
  1. Fetch `/meta/build` with `cache: "no-store"`
  2. Compare to stored build id
  3. On mismatch: clear Cache Storage, unregister service workers, hard-reload once
- Deploy: set **`APP_BUILD_ID`** on backend = **`VITE_APP_BUILD_ID`** on frontend build

---

## 5. Router & main.py integration

### Guest-aware AI endpoints

| Area | File | Guest changes |
|------|------|---------------|
| Studio autogen | `app/main.py` `/autogenerate-images` | Cookie/header session, fingerprint, Turnstile, guest limits, consent audit, SSE error logging |
| Chat | `app/routers/chat.py` | `require_guest_session_id`, fingerprint identity, Turnstile, increment chat on SSE `done` only |
| Multi-image | `app/routers/multi_image_gen.py` | Guest session commit before SSE worker; worker reloads `GuestSession` in own DB session |
| Voice | `app/routers/voice.py` | Cookie/header/cookie WS resolution, Turnstile query param, voice turn increment on `turn_complete` |

### CORS & security (`app/config.py`, `app/main.py`)

- Production CORS: explicit `CORS_ALLOW_ORIGINS` required (empty = block all cross-origin)
- Dev defaults include ports 3000, 4173, 5173, **5174**, 5175 (Ducon Library Vite `strictPort`)
- `allow_credentials=True` — wildcard origins removed for production safety

### Search endpoint shape

- `/search` returns `{ "results": [...] }` (single list) via unified `ducon_vector_store` + `GeminiEmbeddingModel`
- Rate-limited and image-size guarded

---

## 6. Admin, analytics & settings

### Role expansion (`app/admin/admin_auth.py`)

- New roles: **`admin`** (full) and **`analytics`** (read-only dashboards)
- Dependencies:
  - `require_admin` — settings, secrets, user CRUD, audit, error archive
  - `require_admin_or_analytics` — admin session bootstrap
  - `require_analytics_jwt_only` — metrics/errors read without admin password re-auth

### Settings catalog (`app/admin/settings_catalog.py`)

- New **Guest Limits** namespace: `GUEST_GEN_LIMIT`, `GUEST_CHAT_LIMIT`, `GUEST_VOICE_LIMIT`, `GUEST_IP_TOTAL_LIMIT`, `GUEST_SUBNET_LIMIT`
- `GEN_EVAL_STRICTNESS` choice: `relaxed` | `strict` (aliases: `less_restrictive`, `more_restrictive`)
- `PROMPT_GEN_MODEL` documented
- Secrets read-only: `GUEST_CLEANUP_SECRET`, R2 keys, etc.

### Error log API (`app/routers/admin.py`)

- List/filter/archive structured errors from `app_error_logs`
- Analytics role can read; admin can archive

### bcrypt audit fix

- **`bcrypt==4.0.1`** pinned in `pyproject.toml` / `requirements.txt` to avoid passlib version warnings (see `tests/test_turnstile_and_bcrypt.py`)

---

## 7. Image generation & QC pipeline

### Generation eval strictness (`app/gemini.py`, `app/prompt_loader.py`)

- **`GEN_EVAL_STRICTNESS`** env/admin setting:
  - **relaxed** (default): trust model verdict; pass/accepted/rejected tiers
  - **strict**: pass/fail only; all critical sections must pass
- New rubric prompt files: `app/prompts/gen-eval-rubric-relaxed.md`, `gen-eval-rubric-strict.md`
- `build_quality_notice()` surfaces non-fatal QC warnings in API responses

### Tool & agent hardening

- **`app/tool_generate_image.py`** — major expansion: role classification, aspect-ratio auto, verify loop, keyword-only thread args (regression-tested)
- **`app/designer_agent.py`** — cross-worker persistence via `designer_jobs` table; input image storage; async blocking audit
- **`app/image_gen_agent.py`**, **`app/chat_agent.py`** — guest tool gating (`start_designer_job` omitted for guests)
- Studio directions & prompt generator session updates

### Designer job lifecycle

- New **`app/designer_cleanup.py`** — retention (`DESIGNER_JOB_RETENTION_DAYS`, default 30), stale running jobs marked failed (`DESIGNER_JOB_STALE_RUNNING_HOURS`, default 2)
- Integrated into `POST /guest/cleanup` cron

---

## 8. Storage & signed URLs

- **`app/storage.py`** — guest generation paths, signed URL helpers, designer input image roundtrip, path-traversal guards
- **`URL_SIGNING_SECRET`** env (defaults to `JWT_SECRET_KEY`) for local-mode guest image URLs

---

## 9. Dev benchmark dashboard (non-production)

- New **`app/benchmark/`** package + **`app/routers/dev_benchmark.py`**
- Mounted only when `DEV_DASHBOARD_ENABLED=1` **and** `ENV != production`
- OpenRouter provider registry, session config, scoped filesystem tool
- **`dev_dashboard/`** Vite app (local port 5175) — **not deployed to VPS**
- **`benchmark_data/`** added to `.gitignore`
- **Designer agent session fixes (2026-07-08):** each new run clears chat/SSE state in the dashboard; localStorage persists form settings only; backend scopes filesystem scratch to `{root}/{job_id}/` per run; **`POST /dev/designer/jobs/{id}/cancel`** stops in-flight jobs (Stop button in Designer Agent + Live tabs)
- **Session config schema (2026-07-08):** `GET /dev/session-config-schema` and embedded `session.schema` in `GET /dev/designer/config` expose field descriptions, min/max limits, and router applicability (OpenRouter compression, Claude compaction beta, Gemini client-only compaction). Designer Agent UI shows/hides fields based on selected designer router and context policy.

---

## 10. Tests & deploy tooling

### New test suite (`tests/`)

| File | Tests | Focus |
|------|-------|-------|
| `test_guest_usage.py` | 17 | Fingerprint-first identity, IP/subnet caps, UUID rotation |
| `test_guest_identity.py` | 15 | IP extraction, subnet keys, composite hash, no cross-UUID merge |
| `test_guest_session_token.py` | 6 | Cookie sign/verify, header precedence |
| `test_user_isolation.py` | 10 | Rate limiter NAT isolation, chat session separation |
| `test_turnstile_and_bcrypt.py` | 5 | Turnstile dev/prod behavior, bcrypt pin |
| `test_cache_flush.py` | 6 | `/meta/build`, HTML no-cache middleware |
| `test_multi_image_endpoint_deploy.py` | 6 | Guest auth, SSE worker session reload |
| `test_tool_generate_image_regressions.py` | 4 | Thread kwargs, image role classification |
| `test_designer_job_cleanup.py` | 8 | Cleanup cron, designer retention |
| `test_designer_job_cross_worker.py` | 4 | Cross-worker job GET/events/cancel |
| `test_designer_slider_and_async_blocking.py` | 11 | Designer input images, async blocking audit |
| `test_chat_tools_deploy.py` | 3 | Guest vs auth tool lists, frontend `useEffect` dep |
| `test_signup_multistep_deploy.py` | 5 | Frontend signup flow (sibling repo file reads) |
| `test_dev_benchmark_provider_and_tools.py` | 26 | Benchmark provider/tools |
| **Total** | **126** | |

- Dev dependencies: `pytest`, `pytest-asyncio`, `httpx` in `[dependency-groups].dev`
- **`scripts/run_deploy_checks.py`** — syntax compile, import sanity, default pytest subset
- **`tests/DEPLOYMENT_ENDPOINT_CHECKLIST.md`** — manual staging checklist

### Run tests

```bash
uv sync --group dev   # or pip install pytest pytest-asyncio httpx
python -m pytest tests/
python scripts/run_deploy_checks.py
```

---

## 11. New environment variables

| Variable | Required (prod) | Default | Purpose |
|----------|-----------------|---------|---------|
| `APP_BUILD_ID` | Recommended | `dev` | Exposed at `GET /meta/build`; match `VITE_APP_BUILD_ID` |
| `GUEST_SESSION_SECRET` | Recommended | `JWT_SECRET_KEY` | HMAC for `ducon_guest_session` cookie |
| `GUEST_GEN_LIMIT` | No | `3` | Per-device generation cap |
| `GUEST_CHAT_LIMIT` | No | `10` | Per-device chat turn cap |
| `GUEST_VOICE_LIMIT` | No | `5` | Per-device voice turn cap |
| `GUEST_IP_TOTAL_LIMIT` | No | `0` | Shared IP cap (0 = off) |
| `GUEST_SUBNET_LIMIT` | No | `0` | Shared /24\|/64 cap (0 = off) |
| `GUEST_CLEANUP_SECRET` | **Yes** (prod cron) | — | `X-Cron-Secret` for `POST /guest/cleanup` |
| `TURNSTILE_SECRET_KEY` | **Yes** (guest AI) | — | Cloudflare Turnstile server verify |
| `TURNSTILE_ENFORCE` | No | `false` | Force Turnstile in local dev |
| `CORS_ALLOW_ORIGINS` | **Yes** (prod) | dev ports | Comma-separated SPA origins |
| `ENV` | **Yes** | `development` | `production` enables fail-closed behavior |
| `DESIGNER_JOB_RETENTION_DAYS` | No | `30` | Terminal designer job row retention |
| `DESIGNER_JOB_STALE_RUNNING_HOURS` | No | `2` | Mark crashed in-flight jobs failed |
| `GEN_EVAL_STRICTNESS` | No | `relaxed` | QC rubric strictness |
| `PROMPT_GEN_MODEL` | No | `gemini-3-flash-preview` | Prompt writer / QC text model |
| `OPENROUTER_API_KEY` | No | — | Dev benchmark only |
| `DEV_DASHBOARD_ENABLED` | No | off | Local benchmark router |

See **`env_template.txt`** for the full annotated list.

---

## 12. Database migration

### Before deploying this build

1. **Backup** production Postgres.
2. Apply schema (idempotent):

   ```bash
   psql "$DATABASE_URL" -f database_schema.sql
   ```

3. Verify guest identity columns (see **[prodtodo.md](prodtodo.md)** for full runbook, verification SQL, rollback).

### New / altered tables (high level)

- **`guest_sessions`** — +5 identity columns, +3 indexes, chat/voice count constraints
- **`designer_jobs`** — cross-worker designer agent state (JSONB events)
- **`app_error_logs`** — structured application errors

`Base.metadata.create_all` on startup creates missing admin/metrics tables but **does not replace** running `database_schema.sql` for production guest column migration.

---

## 13. Breaking changes & migration notes

| Change | Impact | Action |
|--------|--------|--------|
| Guest session cookie flow | Browsers should call `POST /guest/session` with `credentials: 'include'` | Frontend rollout can trail backend; header still works |
| `X-Guest-Fingerprint` | Quota binds to device fingerprint when sent | Integrate `guestFingerprint.ts`; install `@fingerprintjs/fingerprintjs` |
| `GUEST_IP_GEN_LIMIT` removed | Replaced by `GUEST_IP_TOTAL_LIMIT` (sum of gen+chat+voice) | Update env/admin settings; keep at `0` for office NAT |
| `/search` response shape | Single `results` array (not `text_result` / `image_result` keys) | Update frontend search result parsing if not already done |
| Production CORS | `allow_origins=["*"]` removed | Set `CORS_ALLOW_ORIGINS` to SPA origin(s) |
| Production Turnstile | Missing secret = bot check fails | Set `TURNSTILE_SECRET_KEY` + frontend site key |
| `POST /guest/cleanup` | Requires secret in production | Set `GUEST_CLEANUP_SECRET`; cron sends `X-Cron-Secret` |
| Chroma embedding migration | `ducon_vector_store` + Gemini embeddings | Re-ingest catalog if upgrading from legacy MiniLM/CLIP stores |
| Cookie `SameSite=none` | Requires HTTPS in production | Ensure TLS termination (Cloudflare/nginx) |

### Non-breaking / backward compatible

- `X-Guest-Session-Id` header still accepted (overrides cookie)
- `GET /guest/gen-count` legacy `{ used, limit }` preserved
- `usage_snapshot` includes legacy generation fields
- Guest identity DB columns are additive (`NULL` on existing rows)

---

## 14. New & updated API endpoints (quick reference)

### New

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/guest/session` | Issue signed guest cookie |
| `GET` | `/meta/build` | Build id for PWA cache flush |

### Updated behavior (guest)

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/autogenerate-images` | Turnstile + fingerprint + cookie/header session |
| `POST` | `/generate-multi-image` | Same; guest session committed pre-SSE |
| `POST` | `/chat/message` | Turnstile; chat counted on SSE done |
| `GET` | `/ws/voice` | `turnstile_token` + cookie/header/`guest_session_id` query |
| `GET` | `/guest/usage` | Per-feature snapshot |
| `POST` | `/guest/cleanup` | + designer job pruning |

### Admin

| Area | Notes |
|------|-------|
| `/admin/errors/*` | Structured error log read/archive |
| `/admin/*` auth | `analytics` role read paths |

---

## 15. Frontend integration (Ducon_Library sibling repo)

Reference implementations live in **`app/static/`** (copy into frontend):

### 1. Guest bootstrap (once per visit)

```javascript
await fetch(`${API}/guest/session`, { method: 'POST', credentials: 'include' });
```

### 2. Fingerprint (once per page load)

```javascript
import { getGuestFingerprint } from '@/lib/guestFingerprint';
const fp = await getGuestFingerprint();
// Attach to all guest API calls:
headers['X-Guest-Fingerprint'] = fp;
```

### 3. Credentialed guest calls

```javascript
await fetch(`${API}/guest/usage`, { credentials: 'include' });
// Omit X-Guest-Session-Id unless overriding cookie during migration
```

### 4. Turnstile on guest AI actions

- Send `cf_turnstile_token` on form posts (`/autogenerate-images`, `/chat/message`, `/generate-multi-image`)
- Voice WS: `?turnstile_token=...&guest_session_id=...` (or rely on cookie)

### 5. Cache flush on landing (before React root)

```javascript
import { flushStaleCache } from '@/utils/flushStaleCache';
await flushStaleCache({ buildId: import.meta.env.VITE_APP_BUILD_ID });
```

### 6. Usage UI

- Prefer `GET /guest/usage` → `{ generations, chat, voice }` each with `{ used, limit }`
- Handle `429` with `detail.code === "GUEST_LIMIT_REACHED"`

### 7. Tests that read frontend sources (when sibling repo present)

- `test_signup_multistep_deploy.py` — progressive signup steps
- `test_chat_tools_deploy.py` — `App.jsx` designer job `useEffect` deps
- `test_cache_flush.py` — bootstrap calls `flushStaleCache`
- `test_designer_slider_and_async_blocking.py` — CSS overflow constraints

---

## 16. Deploy checklist (condensed)

See also **VPS_DEPLOYMENT.md** §10 and **[prodtodo.md](prodtodo.md)**.

- [ ] Run `database_schema.sql` on production Postgres (guest identity columns)
- [ ] Set `ENV=production`, strong `JWT_SECRET_KEY`, `DATABASE_URL`
- [ ] Set `GUEST_SESSION_SECRET` (or confirm JWT secret reuse policy)
- [ ] Set `CORS_ALLOW_ORIGINS` to SPA origin(s)
- [ ] Set `TURNSTILE_SECRET_KEY` (+ frontend `VITE_TURNSTILE_SITE_KEY`)
- [ ] Set `GUEST_CLEANUP_SECRET` + schedule `POST /guest/cleanup` cron
- [ ] Set `APP_BUILD_ID` = frontend `VITE_APP_BUILD_ID`
- [ ] Confirm `GUEST_IP_TOTAL_LIMIT=0` and `GUEST_SUBNET_LIMIT=0` unless public abuse caps intended
- [ ] `pip install -r requirements.txt` / `uv sync` (includes `bcrypt==4.0.1`)
- [ ] `python scripts/run_deploy_checks.py` or `pytest tests/`
- [ ] nginx/Caddy: `/ws/` upgrade, SSE no-buffer, `/meta` proxied
- [ ] Smoke: `POST /guest/session` → cookie → `GET /guest/usage` → `GET /meta/build`
- [ ] Frontend: deploy fingerprint + cache flush + credentialed fetch

---

## 17. Files changed (complete list)

### Modified (tracked)

`.gitignore`, `VPS_DEPLOYMENT.md`, `app.zip`, `app/admin/*`, `app/chat_agent.py`, `app/config.py`, `app/db/models.py`, `app/designer_agent.py`, `app/gemini.py`, `app/guest_usage.py`, `app/image_gen_agent.py`, `app/llm_provider.py`, `app/main.py`, `app/otp_service.py`, `app/prompt_generator_session.py`, `app/prompt_loader.py`, `app/prompts/*`, `app/rate_limiter.py`, `app/routers/*`, `app/storage.py`, `app/studio_directions_agent.py`, `app/tool_generate_image.py`, `database_schema.sql`, `env_template.txt`, `pyproject.toml`, `requirements.txt`, `scripts/migrate_admin_otp.py`, `uv.lock`

### Dev benchmark dashboard (2026-07-08)

- **Session config visibility** — Provider-only fields (Claude compaction, OpenRouter compression) now hide correctly per designer router; fixed `visible()` defaulting to show when schema lookup failed, added `applicability` fallback, and derive router from `modelPairId` when pair list is stale.
- **Use max** — Each numeric session/context field has a **Use max** button that applies the schema maximum.
- **Claude compaction API** — `context_management` is sent only on `client.beta.messages.create` (beta `compact-2026-01-12`); stable `messages.create` fallback no longer receives the unsupported kwarg.
- **SESSION_LIMITS** — Updated from provider docs: Gemini 1M context / 65k output, Claude compaction trigger default 150k (min 50k, max 1M), context budget cap 1,048,576.


`app/benchmark/`, `app/build_meta.py`, `app/designer_cleanup.py`, `app/error_logger.py`, `app/guest_identity.py`, `app/guest_session_token.py`, `app/middleware/`, `app/prompts/gen-eval-rubric-*.md`, `app/routers/dev_benchmark.py`, `app/routers/meta.py`, `app/static/flushStaleCache.ts`, `app/static/guestFingerprint.ts`, `dev_dashboard/`, `prodtodo.md`, `scripts/run_deploy_checks.py`, `tests/`

---

## 18. Related documentation

| Document | Purpose |
|----------|---------|
| [prodtodo.md](prodtodo.md) | Production DB migration for guest identity columns |
| [env_template.txt](env_template.txt) | Full environment variable reference |
| [VPS_DEPLOYMENT.md](VPS_DEPLOYMENT.md) | nginx/Caddy, SSE/WS, systemd |
| [tests/DEPLOYMENT_ENDPOINT_CHECKLIST.md](tests/DEPLOYMENT_ENDPOINT_CHECKLIST.md) | Automated vs manual pre-deploy checks |

---

*Generated 2026-07-08 from repository working tree analysis. Not committed unless explicitly requested.*
