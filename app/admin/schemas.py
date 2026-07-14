"""Pydantic schemas for the admin API."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class AdminVerifyRequest(BaseModel):
    password: str


class AdminVerifyResponse(BaseModel):
    admin_session_token: str
    expires_in_minutes: int


class SettingItem(BaseModel):
    namespace: str
    key: str
    value: Any
    # only for non-secret editable settings; secrets are not updatable here


class SettingsUpdateRequest(BaseModel):
    settings: list[SettingItem]


class SettingsUpdateResponse(BaseModel):
    updated: list[str]
    errors: list[dict] = Field(default_factory=list)


class UserPatchRequest(BaseModel):
    role: Optional[str] = None  # customer | admin | analytics | analytics
    banned: Optional[bool] = None  # reserved for future ban table


class AdminLogRequest(BaseModel):
    action: str
    target: Optional[str] = None
    details: Optional[Any] = None
