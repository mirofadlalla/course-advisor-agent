"""
services/chat_service.py — Chat Service (with Metrics Instrumentation)

RESPONSIBILITY: Bridge between the FastAPI endpoint and the PydanticAI agent.
Also instruments every request with:
    - agent_process_ms  : wall-clock time inside run_sync()
    - tokens_in / out   : from result.usage() — actual LLM token counts
    - success / error   : written to MetricsStore via monitoring.store

STREAMING (astream):
    Hybrid agent.iter() driver:
      • Tool-selection model turn  → non-streaming Groq request (groq_compat works)
      • Tool execution             → CallToolsNode stream for status events
      • Final-answer model turn    → streaming PartDeltaEvent tokens
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from pydantic_ai.messages import (
    FunctionToolResultEvent,
    FinalResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_graph import End

from app.agent import create_agent
from app.dependencies import AgentDependencies

logger = logging.getLogger(__name__)

# Groq pricing constants (duplicated here for the return payload)
_COST_PER_1M_IN = 0.59
_COST_PER_1M_OUT = 0.79

_MAX_TOOL_USE_RETRIES = 3
_RETRY_BACKOFF_S = 0.5

_STREAM_END = object()

_TOOL_STATUS_START: dict[str, str] = {
    "search_knowledge": "Searching knowledge base...",
    "get_course_by_name": "Looking up course...",
}
_TOOL_STATUS_DONE: dict[str, str] = {
    "search_knowledge": "✓ Found relevant results",
    "get_course_by_name": "✓ Found matching course",
}


def _is_tool_use_failed(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "tool_use_failed" in msg
        or "Failed to call a function" in msg
        or "tool call validation failed" in msg
    )


def _usage_cost(tokens_in: int, tokens_out: int) -> float:
    return round(
        (tokens_in / 1_000_000) * _COST_PER_1M_IN
        + (tokens_out / 1_000_000) * _COST_PER_1M_OUT,
        8,
    )


def _extract_usage(result: Any) -> tuple[int, int]:
    tokens_in = 0
    tokens_out = 0
    try:
        usage = result.usage()
        if usage is not None:
            tokens_in = getattr(usage, "request_tokens", 0) or 0
            tokens_out = getattr(usage, "response_tokens", 0) or 0
    except Exception as usage_err:
        logger.debug(f"Could not extract usage: {usage_err}")
    return tokens_in, tokens_out


class ChatService:
    """Thin orchestrator between FastAPI and the PydanticAI agent."""

    def __init__(self) -> None:
        self.agent = create_agent()
        logger.info("ChatService initialized.")

    async def _emit_text_stream_events(
        self,
        stream: AsyncIterator[Any],
        queue: asyncio.Queue[Any],
    ) -> None:
        """Forward final-answer text deltas from a model stream to the SSE queue."""
        in_final_answer = False
        streaming_started = False

        async for event in stream:
            if isinstance(event, FinalResultEvent):
                in_final_answer = True
            elif in_final_answer and isinstance(event, PartStartEvent):
                if isinstance(event.part, TextPart) and event.part.content:
                    if not streaming_started:
                        await queue.put(
                            {"type": "status", "text": "Streaming response..."}
                        )
                        streaming_started = True
                    await queue.put(
                        {"type": "token", "text": event.part.content}
                    )
            elif in_final_answer and isinstance(event, PartDeltaEvent):
                if isinstance(event.delta, TextPartDelta):
                    delta = event.delta.content_delta
                    if delta:
                        if not streaming_started:
                            await queue.put(
                                {"type": "status", "text": "Streaming response..."}
                            )
                            streaming_started = True
                        await queue.put({"type": "token", "text": delta})

    async def _drive_iter_with_streaming(
        self,
        question: str,
        deps: AgentDependencies,
        queue: asyncio.Queue[Any],
    ) -> Any:
        """
        Run the agent graph with hybrid streaming.

        Tool-selection uses agent_run.next() (non-streaming Groq). Only the
        final model turn uses node.stream() for native token delivery.
        """
        tools_executed = False

        async with self.agent.iter(question, deps=deps) as agent_run:
            node = agent_run.next_node

            while not isinstance(node, End):
                if self.agent.is_call_tools_node(node):
                    for tool_call in node.model_response.tool_calls:
                        text = _TOOL_STATUS_START.get(tool_call.tool_name)
                        if text:
                            await queue.put({"type": "status", "text": text})

                    async def tool_step(n: Any) -> Any:
                        async with n.stream(agent_run.ctx) as tool_stream:
                            async for event in tool_stream:
                                if isinstance(event, FunctionToolResultEvent):
                                    tool_name = getattr(
                                        event.part, "tool_name", ""
                                    )
                                    done_text = _TOOL_STATUS_DONE.get(tool_name)
                                    if done_text:
                                        await queue.put(
                                            {"type": "status", "text": done_text}
                                        )
                        return await agent_run._advance_graph(n)  # pyright: ignore[reportPrivateUsage]

                    node = await agent_run._run_node_with_hooks(node, tool_step)  # pyright: ignore[reportPrivateUsage]
                    tools_executed = True

                elif self.agent.is_model_request_node(node) and tools_executed:
                    await queue.put(
                        {"type": "status", "text": "Generating answer..."}
                    )

                    async def final_step(n: Any) -> Any:
                        async with n.stream(agent_run.ctx) as model_stream:
                            await self._emit_text_stream_events(model_stream, queue)
                        return await agent_run._advance_graph(n)  # pyright: ignore[reportPrivateUsage]

                    node = await agent_run._run_node_with_hooks(node, final_step)  # pyright: ignore[reportPrivateUsage]

                else:
                    node = await agent_run.next(node)

            result = agent_run.result
            if result is None:
                raise RuntimeError("Agent run finished without a result")
            return result

    async def astream(
        self, question: str, deps: AgentDependencies
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream agent progress and native LLM tokens over SSE."""
        logger.info(f"ChatService.astream: question='{question[:80]}'")

        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def producer() -> None:
            t0 = time.perf_counter()
            phase = "tool"
            last_exc: Exception | None = None

            try:
                for attempt in range(1, _MAX_TOOL_USE_RETRIES + 1):
                    try:
                        result = await self._drive_iter_with_streaming(
                            question, deps, queue
                        )
                        phase = "generation"
                        agent_process_ms = round(
                            (time.perf_counter() - t0) * 1000, 2
                        )
                        tokens_in, tokens_out = _extract_usage(result)
                        await queue.put(
                            {
                                "type": "done",
                                "latency_ms": agent_process_ms,
                                "agent_process_ms": agent_process_ms,
                                "tokens_in": tokens_in,
                                "tokens_out": tokens_out,
                                "cost_usd": _usage_cost(tokens_in, tokens_out),
                            }
                        )
                        return

                    except Exception as exc:
                        last_exc = exc
                        if (
                            _is_tool_use_failed(exc)
                            and attempt < _MAX_TOOL_USE_RETRIES
                        ):
                            logger.warning(
                                f"ChatService.astream: Groq tool_use_failed on "
                                f"attempt {attempt}/{_MAX_TOOL_USE_RETRIES}, "
                                f"retrying in {_RETRY_BACKOFF_S}s … ({exc!r})"
                            )
                            await asyncio.sleep(_RETRY_BACKOFF_S)
                            continue
                        raise

                raise RuntimeError("astream retry loop exhausted") from last_exc

            except Exception as exc:
                agent_process_ms = round((time.perf_counter() - t0) * 1000, 2)
                logger.error(f"ChatService.astream failed: {exc}")
                message = (
                    "Model generation failed."
                    if phase == "generation"
                    else "Knowledge search failed."
                )
                await queue.put(
                    {
                        "type": "error",
                        "message": message,
                        "agent_process_ms": agent_process_ms,
                    }
                )
            finally:
                await queue.put(_STREAM_END)

        task = asyncio.create_task(producer())
        try:
            while True:
                item = await queue.get()
                if item is _STREAM_END:
                    break
                yield item
        finally:
            await task

    async def achat(self, question: str, deps: AgentDependencies) -> dict:
        """Async non-streaming chat — uses agent.run() on the main event loop."""
        logger.info(f"ChatService.achat: question='{question[:80]}'")

        t0 = time.perf_counter()
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_TOOL_USE_RETRIES + 1):
            try:
                result = await self.agent.run(question, deps=deps)
                agent_process_ms = (time.perf_counter() - t0) * 1000
                tokens_in, tokens_out = _extract_usage(result)

                return {
                    "response": result.output,
                    "agent_process_ms": round(agent_process_ms, 2),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "cost_usd": _usage_cost(tokens_in, tokens_out),
                    "success": True,
                    "error_msg": None,
                }

            except Exception as exc:
                last_exc = exc
                if _is_tool_use_failed(exc) and attempt < _MAX_TOOL_USE_RETRIES:
                    logger.warning(
                        f"ChatService.achat: Groq tool_use_failed on attempt "
                        f"{attempt}/{_MAX_TOOL_USE_RETRIES}, retrying in "
                        f"{_RETRY_BACKOFF_S}s … ({exc!r})"
                    )
                    await asyncio.sleep(_RETRY_BACKOFF_S)
                    continue
                logger.error(f"ChatService.achat failed (attempt {attempt}): {exc}")
                raise

        raise RuntimeError("achat retry loop exhausted") from last_exc

    def chat(self, question: str, deps: AgentDependencies) -> dict:
        """Sync chat — uses run_sync() from Starlette's thread pool."""
        logger.info(f"ChatService.chat: question='{question[:80]}'")

        t0 = time.perf_counter()

        try:
            result = self.agent.run_sync(question, deps=deps)
            agent_process_ms = (time.perf_counter() - t0) * 1000
            tokens_in, tokens_out = _extract_usage(result)

            return {
                "response": result.output,
                "agent_process_ms": round(agent_process_ms, 2),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": _usage_cost(tokens_in, tokens_out),
                "success": True,
                "error_msg": None,
            }

        except Exception as exc:
            logger.error(f"ChatService.chat failed: {exc}")
            raise
