from database import Base
from sqlalchemy import select, update, delete, Column, BigInteger, String, Boolean, TIMESTAMP, ForeignKey, UniqueConstraint
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func



class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)
    password_hash = Column(String, nullable=True)
    role = Column(String, nullable=False, server_default='customer')
    google_id = Column(String, unique=True, nullable=True)
    user_consent = Column(Boolean, nullable=False, server_default=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

class Bookmark(Base):
    __tablename__ = "bookmarks"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    image_id = Column(BigInteger, ForeignKey("images.id", ondelete="CASCADE"), nullable=True)

    __table_args__ = (
        UniqueConstraint(user_id, image_id, name="uq_user_bookmark")
    )
    

class Generation(Base):
    __tablename__ = "generations"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    generation_name = Column(String, nullable=True)
    url = Column(String, nullable=False, unique=True) 
    generated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    ducon_image_id = Column(BigInteger, ForeignKey("images.id" ,ondelete="SET NULL"))

class Image(Base):
    __tablename__ = "images"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String)
    filename = Column(String, nullable=False)
    url = Column(String, nullable=False, unique=True)
    uploaded_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

