from app.db.database import Base
from sqlalchemy import Column, BigInteger, Integer, String, Boolean, TIMESTAMP, ForeignKey, UniqueConstraint
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
    user_consent = Column(Boolean, nullable=False, server_default='false')
    marketing_consent = Column(Boolean, nullable=False, server_default='false')
    phone_number = Column(String, nullable=True)
    whatsapp_sms_consent = Column(Boolean, nullable=False, server_default='false')
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    bookmarks = relationship("Bookmark", back_populates="user", cascade="all, delete")
    generations = relationship("Generation", back_populates="user", cascade="all, delete")


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
    generated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    ducon_image_id = Column(BigInteger, ForeignKey("images.id", ondelete="SET NULL"), nullable=True)

    user = relationship("User", back_populates="generations")
    ducon_image = relationship("Image", back_populates="generations")


# ── Guest tables (kept fully separate from authenticated user tables) ──────────

class GuestSession(Base):
    """Tracks a guest browser session and its generation usage for rate limiting."""
    __tablename__ = "guest_sessions"

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id       = Column(String, nullable=False, unique=True, index=True)  # UUID from X-Guest-Session-Id
    ip_hash          = Column(String, nullable=True)                            # SHA-256 of client IP (non-reversible)
    generation_count = Column(Integer, nullable=False, server_default='0')
    created_at       = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    last_used_at     = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    generations = relationship("GuestGeneration", back_populates="guest_session",
                               foreign_keys="GuestGeneration.guest_session_id", passive_deletes=True)


class GuestGeneration(Base):
    """AI generation created by a guest. Expires 48 h after creation unless claimed."""
    __tablename__ = "guest_generations"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    guest_session_id = Column(String, ForeignKey("guest_sessions.session_id", ondelete="SET NULL"), nullable=True)
    generation_name = Column(String, nullable=False)
    url             = Column(String, nullable=False, unique=True)   # R2 key or local path
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

