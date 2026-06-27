"""
services/chat_service.py — Chat Service (with Metrics Instrumentation)

Bridge between FastAPI and PydanticAI with:
  - Conversation persistence
  - Usage logging
  - Behavior tracing
  - Request cancellation per conversation
  - Centralized cost calculation
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from pydantic_ai import UsageLimits
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    FinalResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    UserPromptPart,
)
from pydantic_graph import End

from app.agent import create_agent
from app.cost.calculator import calculate_llm_cost
from app.dependencies import AgentDependencies
from app.repositories.crm_repository import InMemoryCrmRepository
from app.repositories.trace_repository import ITraceRepository
from app.sales.intent import build_intent_instructions, detect_intent
from app.services.cancellation_manager import CancellationManager
from app.services.conversation_service import ConversationService
from app.services.lead_service import LeadService
from app.services.session_store import SessionStore
from app.services.usage_service import UsageService
from app.tracing.collector import TraceCollector, get_trace_collector, set_trace_collector

logger = logging.getLogger(__name__)

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


@dataclass
class TurnContext:
    user_id: str
    conversation_id: str
    message_id: str
    session_key: str


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def _rate_limit_user_message(language_hint: str = "") -> str:
    if language_hint == "ar" or any("\u0600" <= c <= "\u06ff" for c in language_hint):
        return (
            "⚠️ الخدمة مشغولة حالياً بسبب ضغط على الخادم. "
            "من فضلك انتظر 10–20 ثانية وحاول مرة أخرى."
        )
    return (
        "⚠️ The AI service is temporarily busy (rate limit). "
        "Please wait 10–20 seconds and try again."
    )


def _is_tool_use_failed(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "tool_use_failed" in msg
        or "Failed to call a function" in msg
        or "tool call validation failed" in msg
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


def _estimate_embedding_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ChatService:
    """Orchestrator between FastAPI and the PydanticAI agent."""

    def __init__(
        self,
        session_store: SessionStore | None = None,
        lead_service: LeadService | None = None,
        conversation_service: ConversationService | None = None,
        usage_service: UsageService | None = None,
        trace_repository: ITraceRepository | None = None,
        cancellation_manager: CancellationManager | None = None,
    ) -> None:
        self.agent = create_agent()
        self.session_store = session_store or SessionStore()
        self.lead_service = lead_service or LeadService(
            crm_repository=InMemoryCrmRepository()
        )
        self.conversation_service = conversation_service
        self.usage_service = usage_service
        self.trace_repository = trace_repository
        self.cancellation_manager = cancellation_manager or CancellationManager()
        logger.info("ChatService initialized.")

    def _resolve_conversation_id(
        self, conversation_id: str | None, session_id: str | None
    ) -> str:
        return conversation_id or session_id or ""

    async def _ensure_history_loaded(self, conversation_id: str) -> None:
        if not self.conversation_service or not conversation_id:
            return
        if self.session_store.get_history(conversation_id):
            return
        messages = await self.conversation_service.get_history(conversation_id)
        if messages:
            self.session_store.load_messages(conversation_id, messages)

    def _prepare_turn(
        self, question: str, session_key: str
    ) -> tuple[list, str, list[str], str]:
        llm_history = self.session_store.get_llm_history(session_key)
        user_history = self.session_store.get_recent_user_messages(session_key)
        intent = detect_intent(question, user_history)
        instructions = build_intent_instructions(intent)
        return llm_history, instructions, user_history, intent.value

    def _usage_limits(self, deps: AgentDependencies) -> UsageLimits:
        s = deps.settings
        return UsageLimits(
            tool_calls_limit=s.agent_tool_calls_limit,
            request_limit=s.agent_request_limit,
        )

    async def _start_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        question: str,
    ) -> TurnContext:
        session_key = conversation_id or "anonymous"
        await self._ensure_history_loaded(session_key)
        message_id = ""
        if self.conversation_service and conversation_id:
            user_msg = await self.conversation_service.save_user_message(
                conversation_id, question
            )
            message_id = user_msg.id
        else:
            import uuid

            message_id = str(uuid.uuid4())
        return TurnContext(
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            session_key=session_key,
        )

    async def _finalize_turn(
        self,
        *,
        turn: TurnContext,
        question: str,
        response: str,
        user_history: list[str],
        intent_value: str,
        instructions: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        model_key: str,
        embedding_tokens: int = 0,
        cancelled: bool = False,
    ) -> dict[str, Any]:
        from app.sales.intent import VisitorIntent

        intent = VisitorIntent(intent_value)
        _, signals = self.lead_service.analyze_turn(
            question, user_history, response
        )
        ticket = await self.lead_service.maybe_create_ticket(
            session_id=turn.session_key,
            message=question,
            history=user_history,
            assistant_reply=response,
            intent=intent,
            signals=signals,
        )

        cost = calculate_llm_cost(model_key, tokens_in, tokens_out)
        total_tokens = tokens_in + tokens_out
        total_cost = cost.total_cost

        if not cancelled:
            self.session_store.append_turn(turn.session_key, question, response)
            if self.conversation_service and turn.conversation_id:
                await self.conversation_service.save_assistant_message(
                    turn.conversation_id,
                    response,
                    tokens=total_tokens,
                    cost=total_cost,
                )

        trace_id: str | None = None
        collector = get_trace_collector()
        if collector:
            collector.set_intent(intent_value, instructions)
            collector.set_response_metrics(
                response=response,
                latency_ms=latency_ms,
                prompt_tokens=tokens_in,
                completion_tokens=tokens_out,
                embedding_tokens=embedding_tokens,
                prompt_cost=cost.prompt_cost,
                completion_cost=cost.completion_cost,
                embedding_cost=0.0,
                total_cost=total_cost,
                provider=cost.provider,
                model=cost.model,
            )
            if cancelled:
                collector.mark_cancelled()
            if self.trace_repository:
                saved = await self.trace_repository.insert(collector.build())
                trace_id = saved.id

        if self.usage_service and not cancelled:
            await self.usage_service.log_model_call(
                user_id=turn.user_id,
                conversation_id=turn.conversation_id or turn.session_key,
                message_id=turn.message_id,
                model_key=model_key,
                prompt_tokens=tokens_in,
                completion_tokens=tokens_out,
                embedding_tokens=embedding_tokens,
                latency_ms=latency_ms,
            )

        return {
            "visitor_intent": intent_value,
            "lead_qualified": signals.is_qualified,
            "ticket_id": ticket.ticket_id if ticket else None,
            "message_id": turn.message_id,
            "trace_id": trace_id,
            "conversation_id": turn.conversation_id or None,
            "cancelled": cancelled,
        }

    def _finalize_turn_sync(self, **kwargs: Any) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
            in_async_context = True
        except RuntimeError:
            in_async_context = False

        if not in_async_context:
            return asyncio.run(self._finalize_turn(**kwargs))

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, self._finalize_turn(**kwargs))
            return future.result()

    async def _emit_text_stream_events(
        self,
        stream: AsyncIterator[Any],
        queue: asyncio.Queue[Any],
    ) -> None:
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
        message_history: list | None = None,
        instructions: str | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> Any:
        tools_executed = False
        pending_tool_calls: dict[str, dict[str, Any]] = {}

        async with self.agent.iter(
            question,
            deps=deps,
            message_history=message_history or None,
            instructions=instructions,
            usage_limits=usage_limits,
        ) as agent_run:
            node = agent_run.next_node

            while not isinstance(node, End):
                if asyncio.current_task() and asyncio.current_task().cancelled():
                    raise asyncio.CancelledError()

                if self.agent.is_call_tools_node(node):
                    for tool_call in node.model_response.tool_calls:
                        text = _TOOL_STATUS_START.get(tool_call.tool_name)
                        if text:
                            await queue.put({"type": "status", "text": text})
                        pending_tool_calls[tool_call.tool_call_id] = {
                            "tool_name": tool_call.tool_name,
                            "arguments": dict(tool_call.args_as_dict()),
                        }

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
                                    collector = get_trace_collector()
                                    if collector:
                                        tc_id = getattr(event, "tool_call_id", "")
                                        info = pending_tool_calls.get(
                                            tc_id,
                                            {"tool_name": tool_name, "arguments": {}},
                                        )
                                        result_content = getattr(
                                            event.part, "content", event.part
                                        )
                                        collector.add_tool_call(
                                            info["tool_name"],
                                            info["arguments"],
                                            result_content,
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
        self,
        question: str,
        deps: AgentDependencies,
        session_id: str | None = None,
        conversation_id: str | None = None,
        user_id: str = "anonymous",
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream agent progress and native LLM tokens over SSE."""
        conv_id = self._resolve_conversation_id(conversation_id, session_id)
        logger.info(
            f"ChatService.astream: user={user_id} conv={conv_id} question='{question[:80]}'"
        )

        turn = await self._start_turn(
            user_id=user_id, conversation_id=conv_id, question=question
        )
        history, instructions, user_history, intent_value = self._prepare_turn(
            question, turn.session_key
        )
        usage_limits = self._usage_limits(deps)
        model_key = deps.settings.model_name
        queue: asyncio.Queue[Any] = asyncio.Queue()

        collector = TraceCollector(
            user_id=user_id,
            conversation_id=turn.conversation_id or turn.session_key,
            message_id=turn.message_id,
            user_prompt=question,
        )
        collector.set_intent(intent_value, instructions)
        set_trace_collector(collector)

        async def producer() -> None:
            t0 = time.perf_counter()
            phase = "tool"
            last_exc: Exception | None = None
            embedding_tokens = _estimate_embedding_tokens(question)

            try:
                for attempt in range(1, _MAX_TOOL_USE_RETRIES + 1):
                    try:
                        result = await self._drive_iter_with_streaming(
                            question,
                            deps,
                            queue,
                            message_history=history,
                            instructions=instructions,
                            usage_limits=usage_limits,
                        )
                        phase = "generation"
                        agent_process_ms = round(
                            (time.perf_counter() - t0) * 1000, 2
                        )
                        tokens_in, tokens_out = _extract_usage(result)
                        cost = calculate_llm_cost(model_key, tokens_in, tokens_out)
                        meta = await self._finalize_turn(
                            turn=turn,
                            question=question,
                            response=result.output,
                            user_history=user_history,
                            intent_value=intent_value,
                            instructions=instructions,
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                            latency_ms=agent_process_ms,
                            model_key=model_key,
                            embedding_tokens=embedding_tokens,
                        )
                        await queue.put(
                            {
                                "type": "done",
                                "latency_ms": agent_process_ms,
                                "agent_process_ms": agent_process_ms,
                                "tokens_in": tokens_in,
                                "tokens_out": tokens_out,
                                "cost_usd": cost.total_cost,
                                **meta,
                            }
                        )
                        return

                    except asyncio.CancelledError:
                        agent_process_ms = round(
                            (time.perf_counter() - t0) * 1000, 2
                        )
                        await self._finalize_turn(
                            turn=turn,
                            question=question,
                            response="",
                            user_history=user_history,
                            intent_value=intent_value,
                            instructions=instructions,
                            tokens_in=0,
                            tokens_out=0,
                            latency_ms=agent_process_ms,
                            model_key=model_key,
                            cancelled=True,
                        )
                        await queue.put({"type": "cancelled", "message": "Request cancelled"})
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

            except asyncio.CancelledError:
                agent_process_ms = round((time.perf_counter() - t0) * 1000, 2)
                await self._finalize_turn(
                    turn=turn,
                    question=question,
                    response="",
                    user_history=user_history,
                    intent_value=intent_value,
                    instructions=instructions,
                    tokens_in=0,
                    tokens_out=0,
                    latency_ms=agent_process_ms,
                    model_key=model_key,
                    cancelled=True,
                )
                await queue.put({"type": "cancelled", "message": "Request cancelled"})

            except Exception as exc:
                agent_process_ms = round((time.perf_counter() - t0) * 1000, 2)
                logger.error(f"ChatService.astream failed: {exc}")
                if _is_rate_limited(exc):
                    message = _rate_limit_user_message(question)
                elif phase == "generation":
                    message = "Model generation failed."
                else:
                    message = "Knowledge search failed."
                await queue.put(
                    {
                        "type": "error",
                        "message": message,
                        "agent_process_ms": agent_process_ms,
                    }
                )
            finally:
                set_trace_collector(None)
                await queue.put(_STREAM_END)

        producer_task = asyncio.create_task(producer())
        cancel_key = conv_id or turn.session_key
        await self.cancellation_manager.register(cancel_key, producer_task)
        try:
            while True:
                item = await queue.get()
                if item is _STREAM_END:
                    break
                yield item
        finally:
            await self.cancellation_manager.unregister(cancel_key, producer_task)
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

    async def achat(
        self,
        question: str,
        deps: AgentDependencies,
        session_id: str | None = None,
        conversation_id: str | None = None,
        user_id: str = "anonymous",
    ) -> dict:
        """Async non-streaming chat."""
        conv_id = self._resolve_conversation_id(conversation_id, session_id)
        turn = await self._start_turn(
            user_id=user_id, conversation_id=conv_id, question=question
        )
        history, instructions, user_history, intent_value = self._prepare_turn(
            question, turn.session_key
        )
        usage_limits = self._usage_limits(deps)
        model_key = deps.settings.model_name
        embedding_tokens = _estimate_embedding_tokens(question)

        collector = TraceCollector(
            user_id=user_id,
            conversation_id=turn.conversation_id or turn.session_key,
            message_id=turn.message_id,
            user_prompt=question,
        )
        collector.set_intent(intent_value, instructions)
        set_trace_collector(collector)

        t0 = time.perf_counter()
        last_exc: Exception | None = None

        try:
            for attempt in range(1, _MAX_TOOL_USE_RETRIES + 1):
                try:
                    result = await self.agent.run(
                        question,
                        deps=deps,
                        message_history=history or None,
                        instructions=instructions,
                        usage_limits=usage_limits,
                    )
                    agent_process_ms = (time.perf_counter() - t0) * 1000
                    tokens_in, tokens_out = _extract_usage(result)
                    cost = calculate_llm_cost(model_key, tokens_in, tokens_out)
                    meta = await self._finalize_turn(
                        turn=turn,
                        question=question,
                        response=result.output,
                        user_history=user_history,
                        intent_value=intent_value,
                        instructions=instructions,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        latency_ms=agent_process_ms,
                        model_key=model_key,
                        embedding_tokens=embedding_tokens,
                    )

                    return {
                        "response": result.output,
                        "agent_process_ms": round(agent_process_ms, 2),
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "cost_usd": cost.total_cost,
                        "success": True,
                        "error_msg": None,
                        **meta,
                    }

                except Exception as exc:
                    last_exc = exc
                    if _is_tool_use_failed(exc) and attempt < _MAX_TOOL_USE_RETRIES:
                        await asyncio.sleep(_RETRY_BACKOFF_S)
                        continue
                    raise

            raise RuntimeError("achat retry loop exhausted") from last_exc
        finally:
            set_trace_collector(None)

    def chat(
        self,
        question: str,
        deps: AgentDependencies,
        session_id: str | None = None,
        conversation_id: str | None = None,
        user_id: str = "anonymous",
    ) -> dict:
        """Sync chat — uses run_sync() from Starlette's thread pool."""
        conv_id = self._resolve_conversation_id(conversation_id, session_id)
        turn_ctx: TurnContext | None = None
        try:
            turn_ctx = asyncio.run(
                self._start_turn(
                    user_id=user_id, conversation_id=conv_id, question=question
                )
            )
        except RuntimeError:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                turn_ctx = executor.submit(
                    asyncio.run,
                    self._start_turn(
                        user_id=user_id, conversation_id=conv_id, question=question
                    ),
                ).result()

        assert turn_ctx is not None
        history, instructions, user_history, intent_value = self._prepare_turn(
            question, turn_ctx.session_key
        )
        usage_limits = self._usage_limits(deps)
        model_key = deps.settings.model_name
        embedding_tokens = _estimate_embedding_tokens(question)
        t0 = time.perf_counter()

        collector = TraceCollector(
            user_id=user_id,
            conversation_id=turn_ctx.conversation_id or turn_ctx.session_key,
            message_id=turn_ctx.message_id,
            user_prompt=question,
        )
        collector.set_intent(intent_value, instructions)
        set_trace_collector(collector)

        try:
            result = self.agent.run_sync(
                question,
                deps=deps,
                message_history=history or None,
                instructions=instructions,
                usage_limits=usage_limits,
            )
            agent_process_ms = (time.perf_counter() - t0) * 1000
            tokens_in, tokens_out = _extract_usage(result)
            cost = calculate_llm_cost(model_key, tokens_in, tokens_out)
            meta = self._finalize_turn_sync(
                turn=turn_ctx,
                question=question,
                response=result.output,
                user_history=user_history,
                intent_value=intent_value,
                instructions=instructions,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=agent_process_ms,
                model_key=model_key,
                embedding_tokens=embedding_tokens,
            )

            return {
                "response": result.output,
                "agent_process_ms": round(agent_process_ms, 2),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost.total_cost,
                "success": True,
                "error_msg": None,
                **meta,
            }

        except Exception as exc:
            logger.error(f"ChatService.chat failed: {exc}")
            raise
        finally:
            set_trace_collector(None)
