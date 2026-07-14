-- =============================================================================
-- Ducon Library — database schema
--
-- This file is fully IDEMPOTENT: every CREATE uses IF NOT EXISTS and every
-- ALTER uses ADD COLUMN IF NOT EXISTS, so it is safe to re-run on a fresh DB
-- OR on an existing DB to bring it up to date. Run it with psql:
--
--   psql "postgresql://postgres:ducondb@localhost/Ducon_Library" -f database_schema.sql
--
-- For an existing DB, the ALTER TABLE block at the bottom adds any missing
-- columns; the CREATE TABLE IF NOT EXISTS blocks add any missing tables
-- (guest tables, designer_jobs, admin/metrics + OTP); CREATE INDEX IF NOT EXISTS
-- adds any missing indexes.
-- =============================================================================


-- ── 0. Pre-migration index cleanup (idempotent; safe on fresh + existing DBs) ──
-- Replace a non-partial ip_hash index if upgrading from an older schema:
DROP INDEX IF EXISTS idx_guest_sessions_ip_hash;
DROP INDEX IF EXISTS idx_guest_sessions_session_id;        -- redundant with UNIQUE(session_id)
DROP INDEX IF EXISTS idx_guest_generations_user_id;        -- replaced by partial index below
DROP INDEX IF EXISTS idx_guest_generations_expires;        -- replaced by partial index below


-- ── 1. Core tables ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  name varchar NOT NULL,
  email varchar NOT NULL UNIQUE,
  password_hash varchar,
  role varchar NOT NULL DEFAULT 'customer',
  google_id varchar UNIQUE,
  email_verified boolean NOT NULL DEFAULT TRUE,
  user_consent boolean NOT NULL DEFAULT FALSE,
  marketing_consent boolean NOT NULL DEFAULT FALSE,
  phone_number varchar,
  whatsapp_sms_consent boolean NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

CREATE TABLE IF NOT EXISTS email_otps (
  id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  email         varchar NOT NULL,
  otp_hash      varchar NOT NULL,
  purpose       varchar(32) NOT NULL,        -- signup | password_reset
  expires_at    TIMESTAMPTZ NOT NULL,
  attempts      integer NOT NULL DEFAULT 0,
  verified_at   TIMESTAMPTZ,
  pending_data  jsonb,                        -- pending signup payload (name, password hash, consent)
  created_at    TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_email_otps_email         ON email_otps (email);
CREATE INDEX IF NOT EXISTS idx_email_otps_email_purpose ON email_otps (email, purpose);

CREATE TABLE IF NOT EXISTS images (
  id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  name        varchar,
  filename    varchar NOT NULL UNIQUE,
  url         varchar NOT NULL UNIQUE,
  uploaded_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

CREATE TABLE IF NOT EXISTS bookmarks (
  id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  user_id  bigint REFERENCES users (id) ON DELETE CASCADE,
  image_id bigint REFERENCES images (id) ON DELETE CASCADE,
  UNIQUE (user_id, image_id)
);

CREATE TABLE IF NOT EXISTS generations (
  id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  user_id        bigint REFERENCES users (id) ON DELETE CASCADE,
  generation_name varchar NOT NULL,
  url            varchar NOT NULL UNIQUE,
  source_image_url varchar NULL,  -- user's original space photo (before)
  generated_at   TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  ducon_image_id bigint REFERENCES images (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_bookmarks_user_id          ON bookmarks(user_id);
CREATE INDEX IF NOT EXISTS idx_bookmarks_image_id         ON bookmarks(image_id);
CREATE INDEX IF NOT EXISTS idx_generations_user_id        ON generations(user_id);
CREATE INDEX IF NOT EXISTS idx_generations_ducon_image_id ON generations(ducon_image_id);
-- users: retention-cohort scans filter by signup week
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users (created_at);
-- generations: admin per-user time-range scans
CREATE INDEX IF NOT EXISTS idx_generations_user_generated_at ON generations (user_id, generated_at);


-- ── 2. Guest tables ──────────────────────────────────────────────────────────
-- Guest usage limits (per session_id UUID from X-Guest-Session-Id):
--   generation_count — one saved pipeline output (internal retries do not count)
--   chat_turn_count  — one /chat/message SSE done (tool_result continuations do not)
--   voice_turn_count — one voice turn_complete
-- Optional IP cap (GUEST_IP_TOTAL_LIMIT > 0) sums all three columns across rows sharing ip_hash.
-- Default 0 = disabled so guests on the same office NAT each get their own session limits.

CREATE TABLE IF NOT EXISTS guest_sessions (
  id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  session_id        varchar NOT NULL UNIQUE,   -- UUID from X-Guest-Session-Id (unique index implicit)
  ip_hash           varchar,                   -- SHA-256 of client IP (non-reversible)
  fingerprint_hash  varchar,                   -- SHA-256 of browser fingerprint sent as X-Guest-Fingerprint
  composite_hash    varchar,                   -- SHA-256 of server-side identity signals (secondary binding)
  subnet_key        varchar,                   -- IPv4 /24 or IPv6 /64 for supplementary abuse tracking
  ja4_fingerprint   varchar,                   -- Cloudflare JA4/JA3 TLS fingerprint (audit)
  asn               varchar,                   -- Cloudflare ASN when forwarded (audit)
  generation_count  integer NOT NULL DEFAULT 0,
  chat_turn_count   integer NOT NULL DEFAULT 0,
  voice_turn_count  integer NOT NULL DEFAULT 0,
  created_at        TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  last_used_at      TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  CONSTRAINT guest_sessions_generation_count_nonneg CHECK (generation_count >= 0),
  CONSTRAINT guest_sessions_chat_turn_count_nonneg  CHECK (chat_turn_count >= 0),
  CONSTRAINT guest_sessions_voice_turn_count_nonneg CHECK (voice_turn_count >= 0)
);
-- fingerprint_hash lookup for UUID-rotation resistance
CREATE INDEX IF NOT EXISTS guest_sessions_fingerprint_hash_idx ON guest_sessions (fingerprint_hash);
-- composite_hash index for audit queries (not used for cross-session quota merge)
CREATE INDEX IF NOT EXISTS guest_sessions_composite_hash_idx ON guest_sessions (composite_hash);
-- subnet_key for optional shared-subnet abuse caps (GUEST_SUBNET_LIMIT > 0)
CREATE INDEX IF NOT EXISTS idx_guest_sessions_subnet_key ON guest_sessions (subnet_key)
  WHERE subnet_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS guest_generations (
  id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  guest_session_id varchar REFERENCES guest_sessions (session_id) ON DELETE SET NULL,
  generation_name  varchar NOT NULL,
  url              varchar NOT NULL UNIQUE,
  source_image_url varchar NULL,
  ducon_image_id   bigint REFERENCES images (id) ON DELETE SET NULL,
  generated_at     TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  expires_at       TIMESTAMPTZ,
  user_id          bigint REFERENCES users (id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS guest_consent_audit (
  id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  session_id_hash varchar NOT NULL,
  ip_hash         varchar,
  event           varchar NOT NULL,
  logged_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- guest_sessions: ip_hash — IP total cap (sum of usage columns WHERE ip_hash = ?)
CREATE INDEX IF NOT EXISTS idx_guest_sessions_ip_hash ON guest_sessions (ip_hash)
  WHERE ip_hash IS NOT NULL;
-- guest_sessions: stale-session / cleanup sweeps by last activity
CREATE INDEX IF NOT EXISTS idx_guest_sessions_last_used_at ON guest_sessions (last_used_at);
-- guest_generations: claim + list by guest session
CREATE INDEX IF NOT EXISTS idx_guest_generations_session ON guest_generations (guest_session_id);
-- guest_generations: claim after login
CREATE INDEX IF NOT EXISTS idx_guest_generations_user_id ON guest_generations (user_id)
  WHERE user_id IS NOT NULL;
-- guest_generations: /guest/cleanup cron (expires_at < now)
CREATE INDEX IF NOT EXISTS idx_guest_generations_expires ON guest_generations (expires_at)
  WHERE expires_at IS NOT NULL;
-- guest_consent_audit: time-range audit queries
CREATE INDEX IF NOT EXISTS idx_guest_consent_audit_logged_at ON guest_consent_audit (logged_at);


-- ── 2b. Designer Agent jobs ─────────────────────────────────────────────────
-- Long-running designer jobs run in-process on one UvicornWorker; state is
-- mirrored here so GET /designer/jobs/{id} and /events work from any worker
-- (the in-memory JOBS registry is per-process). Events are stored as a JSONB
-- array on this row (append-only via SQL || in app/designer_agent.py).
-- Retention: POST /guest/cleanup (cron) deletes terminal rows older than
-- DESIGNER_JOB_RETENTION_DAYS and marks stale running jobs failed.
CREATE TABLE IF NOT EXISTS designer_jobs (
  id               varchar PRIMARY KEY NOT NULL,        -- uuid4 hex (no dashes)
  user_id          bigint NOT NULL REFERENCES users (id) ON DELETE CASCADE,
  status           varchar NOT NULL DEFAULT 'queued',   -- queued|running|completed|failed|cancelled
  created_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  updated_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  error            text,
  final            jsonb,                               -- terminal payload (best generation, refs, summary)
  events           jsonb NOT NULL DEFAULT '[]'::jsonb,  -- [{type, ...}, ...] SSE replay log
  cancel_requested boolean NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_designer_jobs_user_id        ON designer_jobs (user_id);
CREATE INDEX IF NOT EXISTS idx_designer_jobs_status_updated ON designer_jobs (status, updated_at);


-- ── 2c. Chat / voice session continuity (multi-worker) ───────────────────────
-- Replaces in-process chat_session dicts so interaction_id + transcript are
-- shared across gunicorn workers (chat ↔ voice seed).
CREATE TABLE IF NOT EXISTS chat_sessions (
  id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  user_id          bigint REFERENCES users (id) ON DELETE CASCADE,
  guest_session_id varchar REFERENCES guest_sessions (session_id) ON DELETE CASCADE,
  interaction_id   varchar(128),
  transcript       jsonb NOT NULL DEFAULT '[]'::jsonb,
  updated_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_sessions_user_id
  ON chat_sessions (user_id) WHERE user_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_sessions_guest
  ON chat_sessions (guest_session_id) WHERE guest_session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at ON chat_sessions (updated_at);


-- ── 2d. Multi-image generation jobs (create + poll/SSE) ──────────────────────
-- Same pattern as designer_jobs: owning worker runs the job; any worker can
-- serve GET /generation/jobs/{id} and poll-based /events from this row.
CREATE TABLE IF NOT EXISTS generation_jobs (
  id               varchar PRIMARY KEY NOT NULL,
  user_id          bigint REFERENCES users (id) ON DELETE CASCADE,
  guest_session_id varchar REFERENCES guest_sessions (session_id) ON DELETE SET NULL,
  status           varchar NOT NULL DEFAULT 'queued',
  created_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  updated_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  error            text,
  final            jsonb,
  events           jsonb NOT NULL DEFAULT '[]'::jsonb,
  request_id       varchar(64),
  cancel_requested boolean NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_generation_jobs_user_id        ON generation_jobs (user_id);
CREATE INDEX IF NOT EXISTS idx_generation_jobs_guest          ON generation_jobs (guest_session_id);
CREATE INDEX IF NOT EXISTS idx_generation_jobs_status_updated ON generation_jobs (status, updated_at);


-- ── 3. Admin / metrics tables ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS app_settings (
  id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  namespace   varchar(64)  NOT NULL,
  key         varchar(128) NOT NULL,
  value       text         NOT NULL,                       -- JSON-encoded
  value_type  varchar(16)  NOT NULL,                       -- string|int|float|bool|json
  is_secret   boolean      NOT NULL DEFAULT FALSE,
  description text,
  updated_by  bigint REFERENCES users (id) ON DELETE SET NULL,
  updated_at  TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  CONSTRAINT uq_app_settings_namespace_key UNIQUE (namespace, key)
);

CREATE TABLE IF NOT EXISTS admin_audit_log (
  id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  admin_user_id bigint NOT NULL REFERENCES users (id) ON DELETE SET NULL,
  action        varchar(64) NOT NULL,
  target        varchar(255),
  details       jsonb,
  ip_address    varchar(64),
  created_at    TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_admin_audit_log_created_at ON admin_audit_log (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_admin_audit_log_admin_user ON admin_audit_log (admin_user_id);

CREATE TABLE IF NOT EXISTS api_usage_events (
  id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  user_id          bigint REFERENCES users (id) ON DELETE SET NULL,
  guest_session_id varchar(64),
  agent            varchar(64)  NOT NULL,
  model            varchar(128) NOT NULL,
  provider         varchar(32)  NOT NULL,
  input_tokens     integer      NOT NULL DEFAULT 0,
  output_tokens    integer      NOT NULL DEFAULT 0,
  image_count      integer      NOT NULL DEFAULT 0,
  cost_usd         numeric(12,6) NOT NULL DEFAULT 0,
  latency_ms       integer,
  status           varchar(16)  NOT NULL DEFAULT 'success',
  error_message    text,
  created_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_created_at       ON api_usage_events (created_at);
CREATE INDEX IF NOT EXISTS idx_usage_user_id         ON api_usage_events (user_id);
CREATE INDEX IF NOT EXISTS idx_usage_agent_model     ON api_usage_events (agent, model);
-- Composite index for per-user time-range scans (admin metrics: list_users,
-- user_detail, retention_cohort).
CREATE INDEX IF NOT EXISTS idx_usage_user_created_at ON api_usage_events (user_id, created_at);

CREATE TABLE IF NOT EXISTS usage_daily_rollup (
  id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  day                 date NOT NULL,
  user_id             bigint REFERENCES users (id) ON DELETE SET NULL,
  agent               varchar(64),
  model               varchar(128),
  total_calls         integer       NOT NULL DEFAULT 0,
  total_input_tokens  bigint        NOT NULL DEFAULT 0,
  total_output_tokens bigint        NOT NULL DEFAULT 0,
  total_image_count   integer       NOT NULL DEFAULT 0,
  total_cost_usd      numeric(14,6) NOT NULL DEFAULT 0,
  CONSTRAINT uq_usage_rollup_day_user_agent_model UNIQUE (day, user_id, agent, model)
);
CREATE INDEX IF NOT EXISTS idx_usage_rollup_day ON usage_daily_rollup (day);

CREATE TABLE IF NOT EXISTS session_activity (
  id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  user_id          bigint REFERENCES users (id) ON DELETE SET NULL,
  guest_session_id varchar(64),
  occurred_at      TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_activity_user_day ON session_activity (user_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_session_activity_occurred ON session_activity (occurred_at);


CREATE TABLE IF NOT EXISTS app_error_logs (
  id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  created_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  severity         varchar(16)  NOT NULL DEFAULT 'error',
  category         varchar(32)  NOT NULL,
  source           varchar(255) NOT NULL,
  user_id          bigint REFERENCES users (id) ON DELETE SET NULL,
  guest_session_id varchar(64),
  endpoint         varchar(255),
  model            varchar(128),
  error_type       varchar(128),
  message          text         NOT NULL,
  detail           jsonb,
  request_id       varchar(64),
  http_status      integer,
  provider_status  integer,
  archived_at      TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_app_error_logs_created_at
  ON app_error_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_app_error_logs_category_created
  ON app_error_logs (category, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_app_error_logs_severity_created
  ON app_error_logs (severity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_app_error_logs_user_id
  ON app_error_logs (user_id)
  WHERE user_id IS NOT NULL;


-- ── 4. Bootstrap privileged users (run once, manually) ─────────────────────
-- Full admin:
--   UPDATE users SET role = 'admin' WHERE email = 'you@example.com';
-- Analytics-only (read metrics/errors, no settings/secrets/user CRUD):
--   UPDATE users SET role = 'analytics' WHERE email = 'analyst@example.com';


-- ── 5. Post-migration ALTERs (idempotent) ─────────────────────────────────────
-- These run AFTER the CREATE TABLE blocks above so they're safe on a fresh DB
-- (the columns already exist via CREATE TABLE → IF NOT EXISTS makes them no-ops)
-- AND on an existing DB (they add any missing columns from older schema versions).
-- `email_verified` defaults to TRUE so all pre-existing users stay verified
-- and can keep logging in (OTP verification only gates NEW signups).

ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified boolean NOT NULL DEFAULT TRUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS user_consent boolean NOT NULL DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS marketing_consent boolean NOT NULL DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS phone_number varchar;
ALTER TABLE users ADD COLUMN IF NOT EXISTS whatsapp_sms_consent boolean NOT NULL DEFAULT FALSE;

-- guest_sessions predates chat/voice limits:
ALTER TABLE guest_sessions ADD COLUMN IF NOT EXISTS chat_turn_count  integer NOT NULL DEFAULT 0;
ALTER TABLE guest_sessions ADD COLUMN IF NOT EXISTS voice_turn_count integer NOT NULL DEFAULT 0;
ALTER TABLE guest_sessions DROP CONSTRAINT IF EXISTS guest_sessions_generation_count_nonneg;
ALTER TABLE guest_sessions ADD CONSTRAINT guest_sessions_generation_count_nonneg CHECK (generation_count >= 0);
ALTER TABLE guest_sessions DROP CONSTRAINT IF EXISTS guest_sessions_chat_turn_count_nonneg;
ALTER TABLE guest_sessions ADD CONSTRAINT guest_sessions_chat_turn_count_nonneg CHECK (chat_turn_count >= 0);
ALTER TABLE guest_sessions DROP CONSTRAINT IF EXISTS guest_sessions_voice_turn_count_nonneg;
ALTER TABLE guest_sessions ADD CONSTRAINT guest_sessions_voice_turn_count_nonneg CHECK (voice_turn_count >= 0);

-- guest_sessions server-side identity signals (audit + optional shared caps):
ALTER TABLE guest_sessions ADD COLUMN IF NOT EXISTS fingerprint_hash  varchar;
ALTER TABLE guest_sessions ADD COLUMN IF NOT EXISTS composite_hash  varchar;
ALTER TABLE guest_sessions ADD COLUMN IF NOT EXISTS subnet_key      varchar;
ALTER TABLE guest_sessions ADD COLUMN IF NOT EXISTS ja4_fingerprint varchar;
ALTER TABLE guest_sessions ADD COLUMN IF NOT EXISTS asn             varchar;
CREATE INDEX IF NOT EXISTS guest_sessions_fingerprint_hash_idx ON guest_sessions (fingerprint_hash);
CREATE INDEX IF NOT EXISTS guest_sessions_composite_hash_idx ON guest_sessions (composite_hash);
CREATE INDEX IF NOT EXISTS idx_guest_sessions_subnet_key ON guest_sessions (subnet_key)
  WHERE subnet_key IS NOT NULL;

-- designer_jobs (cross-worker persistence; safe on fresh DB — table created in §2b):
ALTER TABLE designer_jobs ADD COLUMN IF NOT EXISTS status           varchar NOT NULL DEFAULT 'queued';
ALTER TABLE designer_jobs ADD COLUMN IF NOT EXISTS created_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL;
ALTER TABLE designer_jobs ADD COLUMN IF NOT EXISTS updated_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL;
ALTER TABLE designer_jobs ADD COLUMN IF NOT EXISTS error            text;
ALTER TABLE designer_jobs ADD COLUMN IF NOT EXISTS final            jsonb;
ALTER TABLE designer_jobs ADD COLUMN IF NOT EXISTS events           jsonb NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE designer_jobs ADD COLUMN IF NOT EXISTS cancel_requested boolean NOT NULL DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS idx_designer_jobs_user_id        ON designer_jobs (user_id);
CREATE INDEX IF NOT EXISTS idx_designer_jobs_status_updated ON designer_jobs (status, updated_at);

-- Tier 3 tables (create_all does not alter existing DBs — run explicitly on prod):
CREATE TABLE IF NOT EXISTS chat_sessions (
  id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  user_id          bigint REFERENCES users (id) ON DELETE CASCADE,
  guest_session_id varchar REFERENCES guest_sessions (session_id) ON DELETE CASCADE,
  interaction_id   varchar(128),
  transcript       jsonb NOT NULL DEFAULT '[]'::jsonb,
  updated_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_sessions_user_id
  ON chat_sessions (user_id) WHERE user_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_sessions_guest
  ON chat_sessions (guest_session_id) WHERE guest_session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at ON chat_sessions (updated_at);

CREATE TABLE IF NOT EXISTS generation_jobs (
  id               varchar PRIMARY KEY NOT NULL,
  user_id          bigint REFERENCES users (id) ON DELETE CASCADE,
  guest_session_id varchar REFERENCES guest_sessions (session_id) ON DELETE SET NULL,
  status           varchar NOT NULL DEFAULT 'queued',
  created_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  updated_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  error            text,
  final            jsonb,
  events           jsonb NOT NULL DEFAULT '[]'::jsonb,
  request_id       varchar(64),
  cancel_requested boolean NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_generation_jobs_user_id        ON generation_jobs (user_id);
CREATE INDEX IF NOT EXISTS idx_generation_jobs_guest          ON generation_jobs (guest_session_id);
CREATE INDEX IF NOT EXISTS idx_generation_jobs_status_updated ON generation_jobs (status, updated_at);
CREATE TABLE IF NOT EXISTS revoked_jtis (
  jti        varchar(64) PRIMARY KEY NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_revoked_jtis_expires_at ON revoked_jtis (expires_at);

