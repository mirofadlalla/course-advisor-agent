"""
services/chat_service.py — Chat Service

RESPONSIBILITY: Bridge between the FastAPI endpoint and the PydanticAI agent.

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
    The response dict shape is unchanged: {"response": str}
"""

import logging

from app.agent import create_agent
from app.dependencies import AgentDependencies

logger = logging.getLogger(__name__)


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
        Run the agent and return the final response as a dict.

        The agent output is `str` — the model's natural-language response
        produced AFTER all tool calls have completed and results fed back.

        Args:
            question: The user's raw message.
            deps:     AgentDependencies from app.state (repositories + settings).

        Returns:
            {"response": str} — ready for FastAPI ChatResponse serialisation.
        """
        logger.info(f"ChatService.chat: question='{question[:80]}'")

        result = self.agent.run_sync(question, deps=deps)

        logger.debug(f"ChatService.chat: messages={result.all_messages()}")

        # result.output is str — return it directly.
        return {"response": result.output}
