"""Usage log domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class UsageLog(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    conversation_id: str
    message_id: str
    provider: str
    model: str
    model_type: str = "llm"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    embedding_tokens: int = 0
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    embedding_cost: float = 0.0
    total_cost: float = 0.0
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
