from pydantic import BaseModel, EmailStr
from datetime import datetime

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True

class Bookmark(BaseModel):
    ducon_image_id: int

class BookmarkResponse(BaseModel):
    id: int
    ducon_image_name: int
    user_id: int

    class Config:
        from_attributes = True


class Generation(BaseModel):
    ducon_image_name: str

class GenerationResponse(BaseModel):
    generation_name: str
    generated_at: datetime
    user_id: int

    class Config:
        from_attributes = True

