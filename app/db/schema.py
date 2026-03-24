from pydantic import BaseModel, EmailStr
from datetime import datetime
from fastapi import UploadFile

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
    image_id: int

class BookmarkResponse(BaseModel):
    id: int
    image_id: int
    user_id: int

    class Config:
        from_attributes = True


class GenerationResponse(BaseModel):
    id: int
    generation_name: str
    url: str
    generated_at: datetime
    

    class Config:
        from_attributes = True

