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

CREATE TABLE guest_sessions (
  id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
  session_id       varchar NOT NULL UNIQUE,
  ip_hash          varchar,
  generation_count integer NOT NULL DEFAULT 0,
  created_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  last_used_at     TIMESTAMPTZ DEFAULT NOW() NOT NULL
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

CREATE INDEX idx_guest_sessions_session_id   ON guest_sessions (session_id);
CREATE INDEX idx_guest_sessions_ip_hash      ON guest_sessions (ip_hash);
CREATE INDEX idx_guest_generations_session   ON guest_generations (guest_session_id);
CREATE INDEX idx_guest_generations_user_id   ON guest_generations (user_id);
CREATE INDEX idx_guest_generations_expires   ON guest_generations (expires_at);

-- ── ALTER queries for existing databases (run these if tables above don't exist yet) ──
-- CREATE TABLE guest_sessions ( ... );   -- run full CREATE above
-- CREATE TABLE guest_generations ( ... );
-- CREATE TABLE guest_consent_audit ( ... );

