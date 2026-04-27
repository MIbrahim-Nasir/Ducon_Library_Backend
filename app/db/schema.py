from pydantic import BaseModel, EmailStr, field_validator, model_validator
from datetime import datetime
from typing import Optional

from app.validators import validate_email_domain, validate_uae_phone


# ── Auth ──────────────────────────────────────────────
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    user_consent: bool = False
    marketing_consent: bool = False
    phone_number: Optional[str] = None
    whatsapp_sms_consent: bool = False

    @field_validator("email", mode="after")
    @classmethod
    def email_trusted_domain(cls, v: str) -> str:
        return validate_email_domain(v)

    @field_validator("phone_number", mode="before")
    @classmethod
    def phone_uae(cls, v):
        if v is None or v == "":
            return None
        return validate_uae_phone(v)

    @field_validator("name", mode="before")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Name cannot be empty.")
        return v.strip()


class GoogleAuthToken(BaseModel):
    token: str                     # Google ID token from the frontend
    user_consent: bool = False
    marketing_consent: bool = False
    phone_number: Optional[str] = None
    whatsapp_sms_consent: bool = False

    @field_validator("phone_number", mode="before")
    @classmethod
    def phone_uae(cls, v):
        if v is None or v == "":
            return None
        return validate_uae_phone(v)


class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    user_id: Optional[int] = None


# ── User ──────────────────────────────────────────────
class ConsentUpdate(BaseModel):
    user_consent: Optional[bool] = None
    marketing_consent: Optional[bool] = None

class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    phone_number: Optional[str] = None
    whatsapp_sms_consent: Optional[bool] = None
    marketing_consent: Optional[bool] = None
    user_consent: Optional[bool] = None

    @field_validator("name", mode="before")
    @classmethod
    def name_not_empty(cls, v):
        if v is not None and not str(v).strip():
            raise ValueError("Name cannot be empty.")
        return v.strip() if v else v

    @field_validator("phone_number", mode="before")
    @classmethod
    def phone_uae(cls, v):
        if v is None or v == "":
            return None
        return validate_uae_phone(v)

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    user_consent: bool
    marketing_consent: bool
    phone_number: Optional[str] = None
    whatsapp_sms_consent: bool
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

