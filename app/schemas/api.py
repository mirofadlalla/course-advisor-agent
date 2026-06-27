"""
schemas/api.py — API Request / Response Schemas

ChatRequest:
    message    — the user's message
    session_id — optional UUID for conversation tracking (used in metrics)

ChatResponse:
    response         — the agent's reply text
    latency_ms       — total wall-clock time for the request (ms)
    agent_process_ms — time inside chat_service.chat() (ms)
    tokens_in        — LLM input tokens consumed
    tokens_out       — LLM output tokens produced
    cost_usd         — estimated cost based on Groq pricing
    request_id       — UUID for correlation with /metrics/requests
"""

from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    latency_ms: float = 0.0
    agent_process_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    request_id: str = ""
    session_id: Optional[str] = None
    visitor_intent: Optional[str] = None
    ticket_id: Optional[str] = None
    lead_qualified: bool = False
