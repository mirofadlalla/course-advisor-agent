"""Trace collector for behavior replay."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from app.prompts import SYSTEM_PROMPT
from app.schemas.trace import ResponseTrace, ToolCallTrace

_trace_collector_var: ContextVar["TraceCollector | None"] = ContextVar(
    "trace_collector", default=None
)


class TraceCollector:
    """Accumulates trace data during a single agent turn."""

    def __init__(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message_id: str,
        user_prompt: str,
    ) -> None:
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.message_id = message_id
        self.user_prompt = user_prompt
        self.intent = ""
        self.injected_instructions = ""
        self.system_prompt = SYSTEM_PROMPT
        self.tool_calls: list[ToolCallTrace] = []
        self.retrieved_sources: list[str] = []
        self.retrieved_chunks: list[dict[str, Any]] = []
        self.llm_response = ""
        self.latency_ms = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.embedding_tokens = 0
        self.prompt_cost = 0.0
        self.completion_cost = 0.0
        self.embedding_cost = 0.0
        self.total_cost = 0.0
        self.provider = ""
        self.model = ""
        self._cancelled = False

    def set_intent(self, intent: str, instructions: str) -> None:
        self.intent = intent
        self.injected_instructions = instructions

    def add_tool_call(
        self, tool_name: str, arguments: dict[str, Any], result: Any
    ) -> None:
        self.tool_calls.append(
            ToolCallTrace(tool_name=tool_name, arguments=arguments, result=result)
        )
        if tool_name == "search_knowledge" and isinstance(result, list):
            for item in result:
                if hasattr(item, "source"):
                    self.retrieved_sources.append(str(item.source))
                    self.retrieved_chunks.append(
                        {
                            "source": item.source,
                            "text": item.text[:500] if hasattr(item, "text") else str(item),
                            "score": getattr(item, "score", None),
                        }
                    )
                elif isinstance(item, dict):
                    self.retrieved_sources.append(str(item.get("source", "")))
                    self.retrieved_chunks.append(item)

    def set_response_metrics(
        self,
        *,
        response: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        embedding_tokens: int,
        prompt_cost: float,
        completion_cost: float,
        embedding_cost: float,
        total_cost: float,
        provider: str,
        model: str,
    ) -> None:
        self.llm_response = response
        self.latency_ms = latency_ms
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.embedding_tokens = embedding_tokens
        self.prompt_cost = prompt_cost
        self.completion_cost = completion_cost
        self.embedding_cost = embedding_cost
        self.total_cost = total_cost
        self.provider = provider
        self.model = model

    def mark_cancelled(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def build(self) -> ResponseTrace:
        return ResponseTrace(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            message_id=self.message_id,
            user_prompt=self.user_prompt,
            intent=self.intent,
            injected_instructions=self.injected_instructions,
            system_prompt=self.system_prompt,
            tool_calls=self.tool_calls,
            retrieved_sources=list(dict.fromkeys(self.retrieved_sources)),
            retrieved_chunks=self.retrieved_chunks,
            llm_response=self.llm_response,
            latency_ms=self.latency_ms,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            embedding_tokens=self.embedding_tokens,
            prompt_cost=self.prompt_cost,
            completion_cost=self.completion_cost,
            embedding_cost=self.embedding_cost,
            total_cost=self.total_cost,
            provider=self.provider,
            model=self.model,
        )


def set_trace_collector(collector: TraceCollector | None) -> None:
    _trace_collector_var.set(collector)


def get_trace_collector() -> TraceCollector | None:
    return _trace_collector_var.get()
