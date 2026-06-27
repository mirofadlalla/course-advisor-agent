"""Conversation domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class Conversation(BaseModel):
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    title: str = "New conversation"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationCreate(BaseModel):
    title: str = "New conversation"


class ConversationPublic(BaseModel):
    conversation_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
