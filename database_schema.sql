-- =============================================================================
-- Ducon Library — database schema
--
-- This file is fully IDEMPOTENT: every CREATE uses IF NOT EXISTS and every
-- ALTER uses ADD COLUMN IF NOT EXISTS, so it is safe to re-run on a fresh DB
-- OR on an existing DB to bring it up to date. Run it with psql:
--
--   psql "postgresql://postgres:ducondb@localhost/Ducon_Library" -f database_schema.sql
--
-- For an existing DB, the ALTER TABLE block at the top adds any missing
-- columns; the CREATE TABLE IF NOT EXISTS blocks add any missing tables
-- (admin/metrics + OTP); CREATE INDEX IF NOT EXISTS adds any missing indexes.
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
-- IP cap (GUEST_IP_TOTAL_LIMIT) sums all three columns across rows sharing ip_hash.

CREATE TABLE IF NOT EXISTS guest_sessions (
  id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  session_id       varchar NOT NULL UNIQUE,   -- UUID from X-Guest-Session-Id (unique index implicit)
  ip_hash          varchar,                   -- SHA-256 of client IP (non-reversible)
  generation_count integer NOT NULL DEFAULT 0,
  chat_turn_count  integer NOT NULL DEFAULT 0,
  voice_turn_count integer NOT NULL DEFAULT 0,
  created_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  last_used_at     TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  CONSTRAINT guest_sessions_generation_count_nonneg CHECK (generation_count >= 0),
  CONSTRAINT guest_sessions_chat_turn_count_nonneg  CHECK (chat_turn_count >= 0),
  CONSTRAINT guest_sessions_voice_turn_count_nonneg CHECK (voice_turn_count >= 0)
);

CREATE TABLE IF NOT EXISTS guest_generations (
  id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  guest_session_id varchar REFERENCES guest_sessions (session_id) ON DELETE SET NULL,
  generation_name  varchar NOT NULL,
  url              varchar NOT NULL UNIQUE,
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


-- ── 4. Bootstrap first admin (run once, manually) ────────────────────────────
-- UPDATE users SET role = 'admin' WHERE email = 'you@example.com';


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
