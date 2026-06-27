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
"""

import asyncio
import logging
import time

from app.agent import create_agent
from app.config import settings as app_settings
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


def _is_tool_use_failed(exc: Exception) -> bool:
    """
    Return True when the exception is the Groq 400 tool_use_failed error.

    PydanticAI wraps the raw httpx/Groq error in various ways depending on
    version; we check the string representation which always contains the
    error code from Groq's JSON body.
    """
    msg = str(exc)
    return "tool_use_failed" in msg or (
        "400" in msg and "Failed to call a function" in msg
    )


class ChatService:
    """
    Thin orchestrator between FastAPI and the PydanticAI agent.

    Instantiated once in lifespan() and stored on app.state.
    All requests share the same agent instance.
    """

    def __init__(self) -> None:
        self.agent = create_agent()
        logger.info("ChatService initialized.")

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

                try:
                    usage = result.usage()
                    if usage is not None:
                        tokens_in = getattr(usage, "request_tokens", 0) or 0
                        tokens_out = getattr(usage, "response_tokens", 0) or 0
                except Exception as usage_err:
                    logger.debug(f"Could not extract usage: {usage_err}")

                cost_usd = (
                    (tokens_in / 1_000_000) * _COST_PER_1M_IN
                    + (tokens_out / 1_000_000) * _COST_PER_1M_OUT
                )

                return {
                    "response": result.output,
                    "agent_process_ms": round(agent_process_ms, 2),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "cost_usd": round(cost_usd, 8),
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
        success = True
        error_msg = None
        tokens_in = 0
        tokens_out = 0

        try:
            result = self.agent.run_sync(question, deps=deps)
            agent_process_ms = (time.perf_counter() - t0) * 1000

            logger.debug(f"ChatService.chat: messages={result.all_messages()}")

            # Extract token usage from PydanticAI result
            try:
                usage = result.usage()
                if usage is not None:
                    tokens_in = getattr(usage, "request_tokens", 0) or 0
                    tokens_out = getattr(usage, "response_tokens", 0) or 0
            except Exception as usage_err:
                logger.debug(f"Could not extract usage: {usage_err}")

            cost_usd = (
                (tokens_in / 1_000_000) * _COST_PER_1M_IN
                + (tokens_out / 1_000_000) * _COST_PER_1M_OUT
            )

            return {
                "response": result.output,
                "agent_process_ms": round(agent_process_ms, 2),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": round(cost_usd, 8),
                "success": True,
                "error_msg": None,
            }

        except Exception as exc:
            agent_process_ms = (time.perf_counter() - t0) * 1000
            success = False
            error_msg = str(exc)
            logger.error(f"ChatService.chat failed: {exc}")
            raise
