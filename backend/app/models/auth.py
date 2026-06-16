"""Pydantic schemas for auth endpoints."""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=150)
    password: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(default=None, description="Frontend session UUID")


class UserOut(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class UserCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=150)
    password: str = Field(..., min_length=4, max_length=128)
    role: str = Field(default="user", pattern="^(admin|user)$")


class UserUpdate(BaseModel):
    is_active: Optional[bool] = None
    role: Optional[str] = Field(default=None, pattern="^(admin|user)$")
    password: Optional[str] = Field(default=None, min_length=4, max_length=128)


class AccessLogOut(BaseModel):
    id: int
    user_id: int
    username: str
    ip_address: str
    city: Optional[str]
    country: Optional[str]
    user_agent: Optional[str]
    logged_in_at: datetime
    address: Optional[str] = None
    location_source: Optional[str] = None


class ActiveSessionOut(BaseModel):
    id: int
    user_id: int
    username: str
    session_id: str
    created_at: datetime
    last_activity_at: Optional[datetime] = None
