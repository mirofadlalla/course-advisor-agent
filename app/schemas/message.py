"""Message domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tokens: int = 0
    cost: float = 0.0


class MessagePublic(BaseModel):
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    timestamp: datetime
    tokens: int = 0
    cost: float = 0.0
