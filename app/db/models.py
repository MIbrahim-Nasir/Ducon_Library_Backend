from app.db.database import Base
from sqlalchemy import (
    Column,
    BigInteger,
    Integer,
    BigInteger as BigInt,
    String,
    Boolean,
    TIMESTAMP,
    Date,
    Float,
    Numeric,
    ForeignKey,
    UniqueConstraint,
    Text,
    Index,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func


class Image(Base):
    __tablename__ = "images"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String, nullable=True)
    filename = Column(String, nullable=False, unique=True)
    url = Column(String, nullable=False, unique=True)
    uploaded_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    bookmarks = relationship("Bookmark", back_populates="image", cascade="all, delete")
    generations = relationship("Generation", back_populates="ducon_image", passive_deletes=True)


class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)
    password_hash = Column(String, nullable=True)
    role = Column(String, nullable=False, server_default='customer')
    google_id = Column(String, unique=True, nullable=True)
    email_verified = Column(Boolean, nullable=False, server_default='true')
    user_consent = Column(Boolean, nullable=False, server_default='false')
    marketing_consent = Column(Boolean, nullable=False, server_default='false')
    phone_number = Column(String, nullable=True)
    whatsapp_sms_consent = Column(Boolean, nullable=False, server_default='false')
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), index=True)

    bookmarks = relationship("Bookmark", back_populates="user", cascade="all, delete")
    generations = relationship("Generation", back_populates="user", cascade="all, delete")


class EmailOtp(Base):
    """One-time codes for signup verification and password reset."""
    __tablename__ = "email_otps"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    email = Column(String, nullable=False, index=True)
    otp_hash = Column(String, nullable=False)
    purpose = Column(String(32), nullable=False)  # signup | password_reset
    expires_at = Column(TIMESTAMP(timezone=True), nullable=False)
    attempts = Column(Integer, nullable=False, server_default='0')
    verified_at = Column(TIMESTAMP(timezone=True), nullable=True)
    pending_data = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_email_otps_email_purpose", "email", "purpose"),
    )


class Bookmark(Base):
    __tablename__ = "bookmarks"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    image_id = Column(BigInteger, ForeignKey("images.id", ondelete="CASCADE"), nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "image_id", name="uq_user_image_bookmark"),
    )

    user = relationship("User", back_populates="bookmarks")
    image = relationship("Image", back_populates="bookmarks")


class Generation(Base):
    __tablename__ = "generations"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    generation_name = Column(String, nullable=False)
    url = Column(String, nullable=False, unique=True)
    # User's original space photo (storage key). Used for authoritative before/after.
    source_image_url = Column(String, nullable=True)
    generated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    ducon_image_id = Column(BigInteger, ForeignKey("images.id", ondelete="SET NULL"), nullable=True)

    user = relationship("User", back_populates="generations")
    ducon_image = relationship("Image", back_populates="generations")

    __table_args__ = (
        Index("idx_generations_user_id", "user_id"),
        Index("idx_generations_user_generated_at", "user_id", "generated_at"),
    )


# ── Guest tables (kept fully separate from authenticated user tables) ──────────

class GuestSession(Base):
    """Tracks a guest browser session and its generation usage for rate limiting.

    Identity resolution order (most stable → least stable):
      1. fingerprint_hash — SHA-256 of browser fingerprint signals sent by client.
         When present, any new session_id from the same device is remapped to the
         existing session so UUID rotation cannot reset quota.
      2. session_id — server-issued cookie or X-Guest-Session-Id header when no
         fingerprint is available.

    composite_hash / subnet_key / ja4 / asn are stored for audit and optional shared
    caps but never merge different session UUIDs into one quota row.
    """
    __tablename__ = "guest_sessions"

    id                = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id        = Column(String, nullable=False, unique=True, index=True)  # UUID from X-Guest-Session-Id
    ip_hash           = Column(String, nullable=True)                            # SHA-256 of client IP (non-reversible)
    fingerprint_hash  = Column(String, nullable=True, index=True)               # SHA-256 of browser fingerprint (X-Guest-Fingerprint)
    composite_hash    = Column(String, nullable=True, index=True)               # SHA-256 of server-side identity signals
    subnet_key        = Column(String, nullable=True, index=True)               # IPv4 /24 or IPv6 /64 (supplementary abuse)
    ja4_fingerprint   = Column(String, nullable=True)                            # Cloudflare JA4/JA3 (audit)
    asn               = Column(String, nullable=True)                            # Cloudflare ASN when forwarded (audit)
    generation_count  = Column(Integer, nullable=False, server_default='0')
    chat_turn_count   = Column(Integer, nullable=False, server_default='0')
    voice_turn_count  = Column(Integer, nullable=False, server_default='0')
    created_at        = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    last_used_at      = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    generations = relationship("GuestGeneration", back_populates="guest_session",
                               foreign_keys="GuestGeneration.guest_session_id", passive_deletes=True)


class GuestGeneration(Base):
    """AI generation created by a guest. Expires 48 h after creation unless claimed."""
    __tablename__ = "guest_generations"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    guest_session_id = Column(String, ForeignKey("guest_sessions.session_id", ondelete="SET NULL"), nullable=True)
    generation_name = Column(String, nullable=False)
    url             = Column(String, nullable=False, unique=True)   # R2 key or local path
    source_image_url = Column(String, nullable=True)  # user's original space photo key
    ducon_image_id  = Column(BigInteger, ForeignKey("images.id", ondelete="SET NULL"), nullable=True)
    generated_at    = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    expires_at      = Column(TIMESTAMP(timezone=True), nullable=True)  # NULL once claimed → permanent
    user_id         = Column(BigInteger, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)  # set on claim

    guest_session = relationship("GuestSession", back_populates="generations",
                                 foreign_keys=[guest_session_id])
    ducon_image   = relationship("Image", foreign_keys=[ducon_image_id])
    user          = relationship("User", foreign_keys=[user_id])


class GuestConsentAudit(Base):
    """Append-only audit log for guest AI generation consent events. No raw PII stored."""
    __tablename__ = "guest_consent_audit"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id_hash = Column(String, nullable=False)   # SHA-256 of session_id
    ip_hash         = Column(String, nullable=True)    # SHA-256 of IP
    event           = Column(String, nullable=False)   # e.g. "ai_generation_guest_consent"
    logged_at       = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())


# ── Admin / metrics tables ────────────────────────────────────────────────────


class AppSetting(Base):
    """Dynamic, admin-tunable runtime configuration. Hot-reloaded via SettingsStore.

    Precedence for any key: app_settings row (here) > os.getenv > code default.
    Secrets (is_secret=True) are read-only in the admin UI, masked by default,
    and never editable through the API. They are surfaced here only so the admin
    can verify presence (e.g. confirm GOOGLE_API_KEY is set) without exposing
    the value.
    """
    __tablename__ = "app_settings"

    id          = Column(BigInteger, primary_key=True, autoincrement=True)
    namespace   = Column(String(64), nullable=False)   # ai_models|thinking|limits|paths|guest|voice|debug|secrets
    key         = Column(String(128), nullable=False)
    value       = Column(Text, nullable=False)          # JSON-encoded scalar/array/object
    value_type  = Column(String(16), nullable=False)    # string|int|float|bool|json
    is_secret   = Column(Boolean, nullable=False, server_default='false')
    description = Column(Text, nullable=True)
    updated_by  = Column(BigInteger, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    updated_at  = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("namespace", "key", name="uq_app_settings_namespace_key"),
    )


class AdminAuditLog(Base):
    """Append-only log of every admin mutation (settings, roles, bans, reveals)."""
    __tablename__ = "admin_audit_log"

    id            = Column(BigInteger, primary_key=True, autoincrement=True)
    admin_user_id = Column(BigInteger, ForeignKey("users.id", ondelete="SET NULL"), nullable=False)
    action        = Column(String(64), nullable=False)  # settings.update|user.role.change|user.ban|secret.reveal|...
    target        = Column(String(255), nullable=True)
    details       = Column(JSONB, nullable=True)
    ip_address    = Column(String(64), nullable=True)
    created_at    = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_admin_audit_log_created_at", "created_at"),
        Index("idx_admin_audit_log_admin_user", "admin_user_id"),
    )


class ApiUsageEvent(Base):
    """One row per AI provider call. Written in batches by UsageRecorder so it
    never blocks the request path. Drives cost/token/generation analytics."""
    __tablename__ = "api_usage_events"

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id          = Column(BigInteger, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    guest_session_id = Column(String(64), nullable=True)
    agent            = Column(String(64), nullable=False)   # chat|designer|live|studio|image_gen|multi_image|quotation|embedding|session
    model            = Column(String(128), nullable=False)
    provider         = Column(String(32), nullable=False)   # gemini|claude|internal
    input_tokens     = Column(Integer, nullable=False, server_default='0')
    output_tokens    = Column(Integer, nullable=False, server_default='0')
    image_count      = Column(Integer, nullable=False, server_default='0')
    cost_usd         = Column(Numeric(12, 6), nullable=False, server_default='0')
    latency_ms       = Column(Integer, nullable=True)
    status           = Column(String(16), nullable=False, server_default='success')
    error_message    = Column(Text, nullable=True)
    created_at       = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), index=True)

    __table_args__ = (
        # Per-user time-range scans (list_users usage IN (...), user_detail,
        # retention_cohort join on user_id + created_at window).
        Index("idx_usage_user_created_at", "user_id", "created_at"),
        # Per-agent / per-model breakdowns.
        Index("idx_usage_agent_model", "agent", "model"),
    )


class UsageDailyRollup(Base):
    """Pre-aggregated daily metrics per (day, user, agent, model). Computed hourly
    from api_usage_events so dashboard queries are fast even at scale."""
    __tablename__ = "usage_daily_rollup"

    id                  = Column(BigInteger, primary_key=True, autoincrement=True)
    day                 = Column(Date, nullable=False)
    user_id             = Column(BigInteger, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    agent               = Column(String(64), nullable=True)
    model               = Column(String(128), nullable=True)
    total_calls         = Column(Integer, nullable=False, server_default='0')
    total_input_tokens  = Column(BigInt, nullable=False, server_default='0')
    total_output_tokens = Column(BigInt, nullable=False, server_default='0')
    total_image_count   = Column(Integer, nullable=False, server_default='0')
    total_cost_usd      = Column(Numeric(14, 6), nullable=False, server_default='0')

    __table_args__ = (
        UniqueConstraint("day", "user_id", "agent", "model", name="uq_usage_rollup_day_user_agent_model"),
    )


class SessionActivity(Base):
    """Lightweight heartbeats for time-spent-on-platform analytics. One row per
    ~60s of active use per user. Enqueued through the same non-blocking usage
    pipeline so it never slows down client requests."""
    __tablename__ = "session_activity"

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id          = Column(BigInteger, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    guest_session_id = Column(String(64), nullable=True)
    occurred_at      = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        # time_spent: per (user, day) minute dedup + retention scans.
        Index("idx_session_activity_user_occurred", "user_id", "occurred_at"),
        # time_spent daily series scans by occurred_at window.
        Index("idx_session_activity_occurred", "occurred_at"),
    )


class DesignerJobRow(Base):
    """Persistent state for a long-running Designer Agent job.

    The job loop runs in-process on whichever gunicorn/UvicornWorker accepted
    the POST /designer/jobs request, and keeps a live ``asyncio.Queue`` for SSE
    streaming on that worker. Other workers cannot see that in-memory registry,
    so state is mirrored here: any worker can serve ``GET /designer/jobs/{id}``
    and a polling-based ``/events`` stream by reading this row. Without this,
    a poll/events request landing on a different worker returns 404.
    """
    __tablename__ = "designer_jobs"

    id               = Column(String, primary_key=True)  # uuid4 hex
    user_id          = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    status           = Column(String, nullable=False, server_default="queued")  # queued|running|completed|failed|cancelled
    created_at       = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at       = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    error            = Column(Text, nullable=True)
    final            = Column(JSONB, nullable=True)
    events           = Column(JSONB, nullable=False, server_default="[]")
    cancel_requested = Column(Boolean, nullable=False, server_default="false")

    __table_args__ = (
        Index("idx_designer_jobs_user_id", "user_id"),
        Index("idx_designer_jobs_status_updated", "status", "updated_at"),
    )


class ChatSessionRow(Base):
    """Postgres-backed chat ↔ voice continuity (multi-worker safe).

    Replaces the former in-memory ``chat_session`` dicts so interaction_id and
    transcript survive across gunicorn workers.
    """
    __tablename__ = "chat_sessions"

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id          = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    guest_session_id = Column(String, ForeignKey("guest_sessions.session_id", ondelete="CASCADE"), nullable=True)
    interaction_id   = Column(String(128), nullable=True)
    transcript       = Column(JSONB, nullable=False, server_default="[]")
    updated_at       = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index(
            "idx_chat_sessions_user_id",
            "user_id",
            unique=True,
            postgresql_where=text("user_id IS NOT NULL"),
        ),
        Index(
            "idx_chat_sessions_guest",
            "guest_session_id",
            unique=True,
            postgresql_where=text("guest_session_id IS NOT NULL"),
        ),
        Index("idx_chat_sessions_updated_at", "updated_at"),
    )


class GenerationJobRow(Base):
    """Durable multi-image generation job (mirrors designer_jobs).

    Lets clients create a job, disconnect, then poll/SSE for result without
    depending on one long-lived request to the owning worker.
    """
    __tablename__ = "generation_jobs"

    id               = Column(String, primary_key=True)  # uuid4 hex
    user_id          = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    guest_session_id = Column(String, ForeignKey("guest_sessions.session_id", ondelete="SET NULL"), nullable=True)
    status           = Column(String, nullable=False, server_default="queued")
    created_at       = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at       = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    error            = Column(Text, nullable=True)
    final            = Column(JSONB, nullable=True)
    events           = Column(JSONB, nullable=False, server_default="[]")
    request_id       = Column(String(64), nullable=True)
    cancel_requested = Column(Boolean, nullable=False, server_default="false")

    __table_args__ = (
        Index("idx_generation_jobs_user_id", "user_id"),
        Index("idx_generation_jobs_guest", "guest_session_id"),
        Index("idx_generation_jobs_status_updated", "status", "updated_at"),
    )


class AppErrorLog(Base):
    """Structured application errors for ops/debugging. Written by ErrorLogger."""
    __tablename__ = "app_error_logs"

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at       = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    severity         = Column(String(16), nullable=False, server_default='error')  # error|warning
    category         = Column(String(32), nullable=False)
    source           = Column(String(255), nullable=False)
    user_id          = Column(BigInteger, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    guest_session_id = Column(String(64), nullable=True)
    endpoint         = Column(String(255), nullable=True)
    model            = Column(String(128), nullable=True)
    error_type       = Column(String(128), nullable=True)
    message          = Column(Text, nullable=False)
    detail           = Column(JSONB, nullable=True)
    request_id       = Column(String(64), nullable=True)
    http_status      = Column(Integer, nullable=True)
    provider_status  = Column(Integer, nullable=True)
    archived_at      = Column(TIMESTAMP(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_app_error_logs_created_at", "created_at"),
        Index("idx_app_error_logs_category_created", "category", "created_at"),
        Index("idx_app_error_logs_severity_created", "severity", "created_at"),
        Index("idx_app_error_logs_user_id", "user_id"),
    )


class RevokedJti(Base):
    """Durable JWT revocation — shared across gunicorn workers.

    In-memory ``_revoked_jtis`` is still checked first for speed; this table is
    the cross-worker source of truth until the token's ``exp``.
    """
    __tablename__ = "revoked_jtis"

    jti        = Column(String(64), primary_key=True)
    expires_at = Column(TIMESTAMP(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_revoked_jtis_expires_at", "expires_at"),
    )


class RateLimitCounter(Base):
    """Fixed-window rate-limit counters — exact limits across gunicorn workers.

    ``key`` is ``{prefix}:{client_ip}[:{identity}]``; ``window_start`` is the
    epoch-aligned start of the current window. Rows are upserted atomically by
    ``app.rate_limiter`` and purged by the periodic cleanup loop.
    """
    __tablename__ = "rate_limit_counters"

    key          = Column(String(255), primary_key=True)
    window_start = Column(Float, nullable=False)
    count        = Column(Integer, nullable=False, default=1)

