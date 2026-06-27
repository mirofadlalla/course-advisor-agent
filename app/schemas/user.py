"""User domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, EmailStr, Field


class UserRole(StrEnum):
    USER = "user"
    ADMIN = "admin"


class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    full_name: str
    email: EmailStr
    password_hash: str
    role: UserRole = UserRole.USER
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_public(self) -> "UserPublic":
        return UserPublic(
            id=self.id,
            full_name=self.full_name,
            email=self.email,
            role=self.role,
            created_at=self.created_at,
        )


class UserPublic(BaseModel):
    id: str
    full_name: str
    email: EmailStr
    role: UserRole
    created_at: datetime


class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: UserRole = UserRole.USER


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    sub: str
    email: str
    role: str
    type: str  # "access" | "refresh"
