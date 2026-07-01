CREATE TABLE users (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  name varchar NOT NULL,
  email varchar NOT NULL UNIQUE,
  password_hash varchar,
  role varchar NOT NULL DEFAULT 'customer',
  google_id varchar UNIQUE,
  user_consent boolean NOT NULL DEFAULT FALSE,
  marketing_consent boolean NOT NULL DEFAULT FALSE,
  phone_number varchar,
  whatsapp_sms_consent boolean NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- ALTER queries for existing tables:
-- ALTER TABLE users ADD COLUMN marketing_consent boolean NOT NULL DEFAULT FALSE;
-- ALTER TABLE users ADD COLUMN phone_number varchar;
-- ALTER TABLE users ADD COLUMN whatsapp_sms_consent boolean NOT NULL DEFAULT FALSE;

CREATE TABLE images (
	id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
	name varchar,
	filename varchar NOT NULL,
	url varchar NOT NULL UNIQUE,
	uploaded_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

CREATE TABLE bookmarks (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  user_id bigint REFERENCES users (id) ON DELETE CASCADE,
  image_id bigint REFERENCES images (id) ON DELETE CASCADE,
  UNIQUE (user_id, image_id)
);

CREATE TABLE generations (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  user_id bigint REFERENCES users (id) ON DELETE CASCADE,
  generation_name varchar NOT NULL,
  url varchar NOT NULL UNIQUE,
  generated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  ducon_image_id bigint REFERENCES images (id) ON DELETE SET NULL
);

CREATE INDEX idx_bookmarks_user_id ON bookmarks(user_id);
CREATE INDEX idx_generations_user_id ON generations(user_id);

CREATE INDEX idx_bookmarks_image_id ON bookmarks(image_id);
CREATE INDEX idx_generations_ducon_image_id ON generations(ducon_image_id);

-- ── Guest tables ───────────────────────────────────────────────────────────────
-- Guest usage limits (per session_id UUID from X-Guest-Session-Id):
--   generation_count — one saved pipeline output (internal retries do not count)
--   chat_turn_count  — one /chat/message SSE done (tool_result continuations do not)
--   voice_turn_count — one voice turn_complete
-- IP cap (GUEST_IP_TOTAL_LIMIT) sums all three columns across rows sharing ip_hash.

CREATE TABLE guest_sessions (
  id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  session_id       varchar NOT NULL UNIQUE,   -- UUID from X-Guest-Session-Id (unique index implicit)
  ip_hash          varchar,                   -- SHA-256 of client IP (non-reversible)
  generation_count integer NOT NULL DEFAULT 0,
  chat_turn_count  integer NOT NULL DEFAULT 0,
  voice_turn_count integer NOT NULL DEFAULT 0,
  created_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  last_used_at     TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  CONSTRAINT guest_sessions_generation_count_nonneg CHECK (generation_count >= 0),
  CONSTRAINT guest_sessions_chat_turn_count_nonneg CHECK (chat_turn_count >= 0),
  CONSTRAINT guest_sessions_voice_turn_count_nonneg CHECK (voice_turn_count >= 0)
);

CREATE TABLE guest_generations (
  id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  guest_session_id varchar REFERENCES guest_sessions (session_id) ON DELETE SET NULL,
  generation_name  varchar NOT NULL,
  url              varchar NOT NULL UNIQUE,
  ducon_image_id   bigint REFERENCES images (id) ON DELETE SET NULL,
  generated_at     TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  expires_at       TIMESTAMPTZ,
  user_id          bigint REFERENCES users (id) ON DELETE SET NULL
);

CREATE TABLE guest_consent_audit (
  id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  session_id_hash varchar NOT NULL,
  ip_hash         varchar,
  event           varchar NOT NULL,
  logged_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- guest_sessions: ip_hash — IP total cap (sum of usage columns WHERE ip_hash = ?)
CREATE INDEX idx_guest_sessions_ip_hash ON guest_sessions (ip_hash)
  WHERE ip_hash IS NOT NULL;

-- guest_sessions: stale-session / cleanup sweeps by last activity
CREATE INDEX idx_guest_sessions_last_used_at ON guest_sessions (last_used_at);

-- guest_generations: claim + list by guest session
CREATE INDEX idx_guest_generations_session ON guest_generations (guest_session_id);

-- guest_generations: claim after login
CREATE INDEX idx_guest_generations_user_id ON guest_generations (user_id)
  WHERE user_id IS NOT NULL;

-- guest_generations: /guest/cleanup cron (expires_at < now)
CREATE INDEX idx_guest_generations_expires ON guest_generations (expires_at)
  WHERE expires_at IS NOT NULL;

-- guest_consent_audit: time-range audit queries
CREATE INDEX idx_guest_consent_audit_logged_at ON guest_consent_audit (logged_at);

-- ── Migrations for existing databases ───────────────────────────────────────────

-- Guest tables missing entirely — run the CREATE TABLE blocks above, then indexes.

-- guest_sessions exists but predates chat/voice limits:
-- ALTER TABLE guest_sessions ADD COLUMN IF NOT EXISTS chat_turn_count integer NOT NULL DEFAULT 0;
-- ALTER TABLE guest_sessions ADD COLUMN IF NOT EXISTS voice_turn_count integer NOT NULL DEFAULT 0;
-- ALTER TABLE guest_sessions DROP CONSTRAINT IF EXISTS guest_sessions_generation_count_nonneg;
-- ALTER TABLE guest_sessions ADD CONSTRAINT guest_sessions_generation_count_nonneg CHECK (generation_count >= 0);
-- ALTER TABLE guest_sessions DROP CONSTRAINT IF EXISTS guest_sessions_chat_turn_count_nonneg;
-- ALTER TABLE guest_sessions ADD CONSTRAINT guest_sessions_chat_turn_count_nonneg CHECK (chat_turn_count >= 0);
-- ALTER TABLE guest_sessions DROP CONSTRAINT IF EXISTS guest_sessions_voice_turn_count_nonneg;
-- ALTER TABLE guest_sessions ADD CONSTRAINT guest_sessions_voice_turn_count_nonneg CHECK (voice_turn_count >= 0);
-- CREATE INDEX IF NOT EXISTS idx_guest_sessions_last_used_at ON guest_sessions (last_used_at);
-- CREATE INDEX IF NOT EXISTS idx_guest_consent_audit_logged_at ON guest_consent_audit (logged_at);
-- DROP INDEX IF EXISTS idx_guest_sessions_session_id;  -- redundant with UNIQUE(session_id)
-- DROP INDEX IF EXISTS idx_guest_generations_user_id;
-- CREATE INDEX IF NOT EXISTS idx_guest_generations_user_id ON guest_generations (user_id) WHERE user_id IS NOT NULL;
-- DROP INDEX IF EXISTS idx_guest_generations_expires;
-- CREATE INDEX IF NOT EXISTS idx_guest_generations_expires ON guest_generations (expires_at) WHERE expires_at IS NOT NULL;

-- Replace non-partial ip_hash index if upgrading from an older schema:
-- DROP INDEX IF EXISTS idx_guest_sessions_ip_hash;
-- CREATE INDEX IF NOT EXISTS idx_guest_sessions_ip_hash ON guest_sessions (ip_hash) WHERE ip_hash IS NOT NULL;

