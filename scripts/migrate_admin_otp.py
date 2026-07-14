"""One-shot idempotent migration for the admin + OTP schema changes.

Run once against an EXISTING database that predates the admin panel / OTP work.
Safe to re-run (everything is IF NOT EXISTS).

Usage:
    .venv\\Scripts\\python.exe scripts\\migrate_admin_otp.py

Reads DATABASE_URL from .env via the app's engine.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure the project root (parent of scripts/) is importable when run directly
# as `python scripts/migrate_admin_otp.py` (which otherwise only puts scripts/
# on sys.path).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("migrate")


# All statements are idempotent (ADD COLUMN IF NOT EXISTS / CREATE TABLE IF NOT
# EXISTS / CREATE INDEX IF NOT EXISTS). Postgres supports ADD COLUMN IF NOT
# EXISTS since 9.6.
from sqlalchemy import text

STATEMENTS: list[str] = [
    # ── OTP / signup verification ────────────────────────────────────────────
    # users.email_verified was added by the OTP work; create_all does NOT alter
    # existing tables, so existing DBs are missing it. Default TRUE so all
    # pre-existing users stay verified and can still log in.
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified boolean NOT NULL DEFAULT TRUE",

    # email_otps table (new in OTP work). create_all usually makes this, but
    # include it here for completeness / fresh-from-scratch safety.
    """
    CREATE TABLE IF NOT EXISTS email_otps (
      id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
      email         varchar NOT NULL,
      otp_hash      varchar NOT NULL,
      purpose       varchar(32) NOT NULL,
      expires_at    TIMESTAMPTZ NOT NULL,
      attempts      integer NOT NULL DEFAULT 0,
      verified_at   TIMESTAMPTZ,
      pending_data  jsonb,
      created_at    TIMESTAMPTZ DEFAULT NOW() NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_email_otps_email ON email_otps (email)",
    "CREATE INDEX IF NOT EXISTS idx_email_otps_email_purpose ON email_otps (email, purpose)",

    # ── Admin / metrics tables ───────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS app_settings (
      id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
      namespace   varchar(64)  NOT NULL,
      key         varchar(128) NOT NULL,
      value       text         NOT NULL,
      value_type  varchar(16)  NOT NULL,
      is_secret   boolean      NOT NULL DEFAULT FALSE,
      description text,
      updated_by  bigint REFERENCES users (id) ON DELETE SET NULL,
      updated_at  TIMESTAMPTZ DEFAULT NOW() NOT NULL,
      CONSTRAINT uq_app_settings_namespace_key UNIQUE (namespace, key)
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS admin_audit_log (
      id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
      admin_user_id bigint NOT NULL REFERENCES users (id) ON DELETE SET NULL,
      action        varchar(64) NOT NULL,
      target        varchar(255),
      details       jsonb,
      ip_address    varchar(64),
      created_at    TIMESTAMPTZ DEFAULT NOW() NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_admin_audit_log_created_at ON admin_audit_log (created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_admin_audit_log_admin_user ON admin_audit_log (admin_user_id)",

    """
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
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_usage_created_at ON api_usage_events (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_usage_user_id ON api_usage_events (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_usage_agent_model ON api_usage_events (agent, model)",
    "CREATE INDEX IF NOT EXISTS idx_usage_user_created_at ON api_usage_events (user_id, created_at)",

    """
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
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_usage_rollup_day ON usage_daily_rollup (day)",

    """
    CREATE TABLE IF NOT EXISTS session_activity (
      id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY NOT NULL,
      user_id          bigint REFERENCES users (id) ON DELETE SET NULL,
      guest_session_id varchar(64),
      occurred_at      TIMESTAMPTZ DEFAULT NOW() NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_session_activity_user_day ON session_activity (user_id, occurred_at)",
    "CREATE INDEX IF NOT EXISTS idx_session_activity_occurred ON session_activity (occurred_at)",

    # Helpful supporting indexes on pre-existing tables (admin metrics hot paths)
    "CREATE INDEX IF NOT EXISTS idx_generations_user_id ON generations (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_generations_user_generated_at ON generations (user_id, generated_at)",
    "CREATE INDEX IF NOT EXISTS idx_users_created_at ON users (created_at)",
]


async def main() -> int:
    from app.db.database import engine

    log.info("Connecting to database via app engine …")
    executed = 0
    skipped_errors = 0
    async with engine.begin() as conn:
        for stmt in STATEMENTS:
            cleaned = " ".join(stmt.split())
            preview = cleaned[:80] + ("…" if len(cleaned) > 80 else "")
            try:
                await conn.execute(text(stmt))
                executed += 1
                log.info("  ✓ %s", preview)
            except Exception as exc:
                # Most "already exists" cases are handled by IF NOT EXISTS, but
                # be defensive: a duplicate index name etc. shouldn't abort the
                # whole migration.
                skipped_errors += 1
                log.warning("  ⚠ skipped: %s — %s", preview, exc.__class__.__name__)

    log.info("Done. Executed %d statement(s); skipped %d with warnings.", executed, skipped_errors)
    log.info("Next steps:")
    log.info("  1. Promote an admin:  UPDATE users SET role='admin' WHERE email='you@example.com';")
    log.info("     Analytics-only:     UPDATE users SET role='analytics' WHERE email='analyst@example.com';")
    log.info("  2. Set ADMIN_PASSWORD_HASH in .env (run: python -m scripts.hash_admin_password)")
    log.info("  3. Set RESEND_API_KEY / RESEND_FROM_EMAIL in .env for OTP emails")
    log.info("  4. Restart uvicorn")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
