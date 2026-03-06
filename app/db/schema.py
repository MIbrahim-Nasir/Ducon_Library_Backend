from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional


# ── Auth ──────────────────────────────────────────────
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    user_consent: bool = False

class GoogleAuthToken(BaseModel):
    token: str                     # Google ID token from the frontend
    user_consent: bool = False

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    user_id: Optional[int] = None


# ── User ──────────────────────────────────────────────
class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    user_consent: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ── Bookmark ──────────────────────────────────────────
class BookmarkCreate(BaseModel):
    image_id: int

class BookmarkResponse(BaseModel):
    id: int
    image_id: int
    user_id: int

    class Config:
        from_attributes = True


# ── Generation ────────────────────────────────────────
class GenerationCreate(BaseModel):
    ducon_image_name: str

class GenerationResponse(BaseModel):
    id: int
    generation_name: str
    url: str              # R2 object key, e.g. "generations/1/file.png"
    signed_url: Optional[str] = None  # freshly generated presigned GET URL
    generated_at: datetime
    user_id: int
    ducon_image_id: Optional[int] = None

    class Config:
        from_attributes = True

