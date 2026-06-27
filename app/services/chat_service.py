"""
<<<<<<< HEAD
services/chat_service.py — Chat Service (with Metrics Instrumentation)

RESPONSIBILITY: Bridge between the FastAPI endpoint and the PydanticAI agent.
Also instruments every request with:
    - agent_process_ms  : wall-clock time inside run_sync()
    - tokens_in / out   : from result.usage() — actual LLM token counts
    - success / error   : written to MetricsStore via monitoring.store
=======
services/chat_service.py — Chat Service

RESPONSIBILITY: Bridge between the FastAPI endpoint and the PydanticAI agent.
>>>>>>> 5e4b5cf90fe050527dad2fc30929cda1b3623f63

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
<<<<<<< HEAD
    The response dict shape is extended with timing + token fields.
"""

import logging
import time

from app.agent import create_agent
from app.config import settings as app_settings
=======
    The response dict shape is unchanged: {"response": str}
"""

import logging

from app.agent import create_agent
>>>>>>> 5e4b5cf90fe050527dad2fc30929cda1b3623f63
from app.dependencies import AgentDependencies

logger = logging.getLogger(__name__)

<<<<<<< HEAD
# Groq pricing constants (duplicated here for the return payload)
_COST_PER_1M_IN = 0.59
_COST_PER_1M_OUT = 0.79

=======
>>>>>>> 5e4b5cf90fe050527dad2fc30929cda1b3623f63

class ChatService:
    """
    Thin orchestrator between FastAPI and the PydanticAI agent.

    Instantiated once in lifespan() and stored on app.state.
    All requests share the same agent instance.
    """

    def __init__(self) -> None:
        self.agent = create_agent()
        logger.info("ChatService initialized.")

    def chat(self, question: str, deps: AgentDependencies) -> dict:
        """
<<<<<<< HEAD
        Run the agent and return the final response with performance metrics.
=======
        Run the agent and return the final response as a dict.
>>>>>>> 5e4b5cf90fe050527dad2fc30929cda1b3623f63

        The agent output is `str` — the model's natural-language response
        produced AFTER all tool calls have completed and results fed back.

        Args:
            question: The user's raw message.
            deps:     AgentDependencies from app.state (repositories + settings).

        Returns:
<<<<<<< HEAD
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
=======
            {"response": str} — ready for FastAPI ChatResponse serialisation.
        """
        logger.info(f"ChatService.chat: question='{question[:80]}'")

        result = self.agent.run_sync(question, deps=deps)

        logger.debug(f"ChatService.chat: messages={result.all_messages()}")

        # result.output is str — return it directly.
        return {"response": result.output}
>>>>>>> 5e4b5cf90fe050527dad2fc30929cda1b3623f63
