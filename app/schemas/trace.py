"""Response trace domain models for behavior replay."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ToolCallTrace(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class ResponseTrace(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    conversation_id: str
    message_id: str
    user_prompt: str
    intent: str = ""
    injected_instructions: str = ""
    system_prompt: str = ""
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    llm_response: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    embedding_cost: float = 0.0
    total_cost: float = 0.0
    provider: str = ""
    model: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_replay_steps(self) -> list[dict[str, Any]]:
        """Ordered steps for debugger-style UI replay."""
        steps: list[dict[str, Any]] = [
            {"step": "User Prompt", "data": self.user_prompt},
            {"step": "Intent", "data": self.intent},
            {"step": "Injected Instructions", "data": self.injected_instructions},
            {"step": "System Prompt", "data": self.system_prompt},
        ]
        for tc in self.tool_calls:
            steps.append({"step": "Tool Call", "data": tc.tool_name})
            steps.append({"step": "Arguments", "data": tc.arguments})
            steps.append({"step": "Results", "data": tc.result})
        steps.extend(
            [
                {"step": "Retrieved Sources", "data": self.retrieved_sources},
                {"step": "Retrieved Chunks", "data": self.retrieved_chunks},
                {"step": "Assistant Reply", "data": self.llm_response},
                {
                    "step": "Tokens",
                    "data": {
                        "prompt": self.prompt_tokens,
                        "completion": self.completion_tokens,
                        "embedding": self.embedding_tokens,
                    },
                },
                {"step": "Latency", "data": f"{self.latency_ms:.2f} ms"},
                {
                    "step": "Cost",
                    "data": {
                        "prompt": self.prompt_cost,
                        "completion": self.completion_cost,
                        "embedding": self.embedding_cost,
                        "total": self.total_cost,
                    },
                },
            ]
        )
        return steps
