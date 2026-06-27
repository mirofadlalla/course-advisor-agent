"""
services/chat_service.py — Chat Service (with Metrics Instrumentation)

RESPONSIBILITY: Bridge between the FastAPI endpoint and the PydanticAI agent.
Also instruments every request with:
    - agent_process_ms  : wall-clock time inside run_sync()
    - tokens_in / out   : from result.usage() — actual LLM token counts
    - success / error   : written to MetricsStore via monitoring.store

CHANGE FROM PREVIOUS VERSION:
    The agent now returns `str` instead of `AgentResponse`.

    Previous (broken):
        result.output → AgentResponse (via output_type= hidden final_result tool)
        Problem: caused Groq parallel-tool-call interference — the model called
        `final_result` in the same turn as a real tool, terminating the agent
        loop before the tool result could be fed back to the model.

    Current (correct):
        result.output → str (PydanticAI default)
        The agent loop terminates only when the model emits plain text with
        no tool calls. Tool results are always fed back correctly first.

    ChatService.chat() now returns result.output directly (it's already a str).
    The response dict shape is extended with timing + token fields.

STREAMING (astream):
    Uses PydanticAI run_stream_events() so tool calls go through agent.run()
    (non-streaming Groq requests where groq_compat recovers malformed tool XML).
    Final-answer tokens arrive as PartDeltaEvent / TextPart deltas in real time.
    Application status events (not chain-of-thought) come from tool call events.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from pydantic_ai import AgentRunResultEvent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    FinalResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)

from app.agent import create_agent
from app.dependencies import AgentDependencies

logger = logging.getLogger(__name__)

# Groq pricing constants (duplicated here for the return payload)
_COST_PER_1M_IN = 0.59
_COST_PER_1M_OUT = 0.79

# Retry configuration for Groq tool_use_failed errors.
# llama-3.3-70b occasionally emits malformed tool-call XML (missing `>`
# after the function name), causing Groq to return HTTP 400 with
# code="tool_use_failed".  The error is non-deterministic — a second
# attempt with the exact same request usually succeeds.
_MAX_TOOL_USE_RETRIES = 3
_RETRY_BACKOFF_S = 0.5   # seconds to wait between retries

_STREAM_END = object()

# Observable application states shown in the thinking timeline (not CoT).
_TOOL_STATUS_START: dict[str, str] = {
    "search_knowledge": "Searching knowledge base...",
    "get_course_by_name": "Looking up course...",
}
_TOOL_STATUS_DONE: dict[str, str] = {
    "search_knowledge": "✓ Found relevant results",
    "get_course_by_name": "✓ Found matching course",
}


def _is_tool_use_failed(exc: Exception) -> bool:
    """
    Return True when the exception is the Groq tool_use_failed error.

    PydanticAI wraps the raw httpx/Groq error in various ways depending on
    version; the message may omit the error code or HTTP status.
    """
    msg = str(exc)
    return "tool_use_failed" in msg or "Failed to call a function" in msg


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
    """
    Thin orchestrator between FastAPI and the PydanticAI agent.

    Instantiated once in lifespan() and stored on app.state.
    All requests share the same agent instance.
    """

    def __init__(self) -> None:
        self.agent = create_agent()
        logger.info("ChatService initialized.")

    async def astream(
        self, question: str, deps: AgentDependencies
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream agent progress and native LLM tokens over SSE.

        Yields dict events:
            {"type": "status", "text": "..."}   — application timeline step
            {"type": "token",  "text": "..."}   — LLM text delta
            {"type": "done",   ...metrics...}   — run complete
            {"type": "error",  "message": "..."} — user-facing failure
        """
        logger.info(f"ChatService.astream: question='{question[:80]}'")

        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def producer() -> None:
            t0 = time.perf_counter()
            phase = "tool"
            last_exc: Exception | None = None

            try:
                for attempt in range(1, _MAX_TOOL_USE_RETRIES + 1):
                    try:
                        # run_stream_events() wraps agent.run() — tool calls use
                        # non-streaming Groq requests where groq_compat can recover
                        # malformed XML tool calls.  run_stream() would hit Groq's
                        # streaming API on the tool turn and fail at peek time
                        # before recovery runs.
                        async with self.agent.run_stream_events(
                            question,
                            deps=deps,
                        ) as events:
                            in_final_answer = False
                            streaming_started = False

                            async for event in events:
                                if isinstance(event, FunctionToolCallEvent):
                                    text = _TOOL_STATUS_START.get(
                                        event.part.tool_name
                                    )
                                    if text:
                                        await queue.put(
                                            {"type": "status", "text": text}
                                        )
                                elif isinstance(event, FunctionToolResultEvent):
                                    tool_name = getattr(
                                        event.part, "tool_name", ""
                                    )
                                    text = _TOOL_STATUS_DONE.get(tool_name)
                                    if text:
                                        await queue.put(
                                            {"type": "status", "text": text}
                                        )
                                elif isinstance(event, FinalResultEvent):
                                    in_final_answer = True
                                    phase = "generation"
                                    await queue.put(
                                        {
                                            "type": "status",
                                            "text": "Generating answer...",
                                        }
                                    )
                                elif in_final_answer and isinstance(
                                    event, PartStartEvent
                                ):
                                    if isinstance(event.part, TextPart):
                                        content = event.part.content
                                        if content:
                                            if not streaming_started:
                                                await queue.put(
                                                    {
                                                        "type": "status",
                                                        "text": "Streaming response...",
                                                    }
                                                )
                                                streaming_started = True
                                            await queue.put(
                                                {"type": "token", "text": content}
                                            )
                                elif in_final_answer and isinstance(
                                    event, PartDeltaEvent
                                ):
                                    if isinstance(event.delta, TextPartDelta):
                                        delta = event.delta.content_delta
                                        if delta:
                                            if not streaming_started:
                                                await queue.put(
                                                    {
                                                        "type": "status",
                                                        "text": "Streaming response...",
                                                    }
                                                )
                                                streaming_started = True
                                            await queue.put(
                                                {"type": "token", "text": delta}
                                            )
                                elif isinstance(event, AgentRunResultEvent):
                                    agent_process_ms = round(
                                        (time.perf_counter() - t0) * 1000, 2
                                    )
                                    tokens_in, tokens_out = _extract_usage(
                                        event.result
                                    )
                                    await queue.put(
                                        {
                                            "type": "done",
                                            "latency_ms": agent_process_ms,
                                            "agent_process_ms": agent_process_ms,
                                            "tokens_in": tokens_in,
                                            "tokens_out": tokens_out,
                                            "cost_usd": _usage_cost(
                                                tokens_in, tokens_out
                                            ),
                                        }
                                    )
                                    return

                        # Iterator ended without AgentRunResultEvent
                        raise RuntimeError(
                            "Agent run finished without a result event"
                        )

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
        """
        Async version of chat(). Use this from async code (e.g. /chat/stream).

        WHY THIS EXISTS (BUG FIX):
            The old pattern was:
                loop.run_in_executor(None, lambda: chat_service.chat(...))
            which calls agent.run_sync() — i.e. loop.run_until_complete() —
            inside a WORKER THREAD with its own freshly-created event loop.

            But self.agent (and the AsyncGroq/httpx.AsyncClient inside it) was
            constructed ONCE in lifespan(), bound to the MAIN uvicorn event loop.
            Reusing that same async HTTP client from a *different* event loop
            (a new one per worker thread) is explicitly unsafe in httpx/asyncio:
            internal locks, connection-pool state, and HTTP/2 framing can be
            corrupted/interleaved across concurrent requests, which is
            consistent with seeing genuine "400 Bad Request" responses come
            back from Groq under concurrent load (multiple users / monitoring
            polling /metrics at the same time) even though the request body
            constructed by pydantic-ai is valid.

            FIX: never hop threads/event loops. Call the agent with `await
            self.agent.run(...)` directly inside the SAME event loop that
            owns the httpx client (the main uvicorn loop). No run_sync, no
            run_in_executor, no thread pool.
        """
        logger.info(f"ChatService.achat: question='{question[:80]}'")

        t0 = time.perf_counter()
        tokens_in = 0
        tokens_out = 0
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_TOOL_USE_RETRIES + 1):
            try:
                result = await self.agent.run(question, deps=deps)
                agent_process_ms = (time.perf_counter() - t0) * 1000

                logger.debug(f"ChatService.achat: messages={result.all_messages()}")

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
                # Non-retryable error or final attempt — propagate.
                agent_process_ms = (time.perf_counter() - t0) * 1000
                logger.error(f"ChatService.achat failed (attempt {attempt}): {exc}")
                raise

        # Should be unreachable, but satisfy the type checker.
        raise RuntimeError("achat retry loop exhausted") from last_exc

    def chat(self, question: str, deps: AgentDependencies) -> dict:
        """
        Run the agent and return the final response with performance metrics.

        The agent output is `str` — the model's natural-language response
        produced AFTER all tool calls have completed and results fed back.

        NOTE: This sync version uses run_sync() and is safe to call from a
        plain sync FastAPI route (Starlette runs those in its own threadpool
        consistently per-request), but must NEVER be combined with
        run_in_executor() from async code that also runs the agent — see
        achat() docstring above for why. Prefer achat() for any new code.

        Args:
            question: The user's raw message.
            deps:     AgentDependencies from app.state (repositories + settings).

        Returns:
            dict with keys:
                response         — agent reply string
                agent_process_ms — time inside run_sync (proxy for TTFT)
                tokens_in        — LLM request tokens
                tokens_out       — LLM response tokens
                cost_usd         — estimated cost (Groq pricing)
                success          — True
        """
        logger.info(f"ChatService.chat: question='{question[:80]}'")

        t0 = time.perf_counter()
        tokens_in = 0
        tokens_out = 0

        try:
            result = self.agent.run_sync(question, deps=deps)
            agent_process_ms = (time.perf_counter() - t0) * 1000

            logger.debug(f"ChatService.chat: messages={result.all_messages()}")

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
