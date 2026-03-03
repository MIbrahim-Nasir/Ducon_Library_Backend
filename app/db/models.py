from app.db.database import Base
from sqlalchemy import Column, BigInteger, String, Boolean, TIMESTAMP, ForeignKey, UniqueConstraint
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

