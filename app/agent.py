"""
agent.py — PydanticAI Agent Factory

See module history in git for the Groq parallel-tool-call fix (output_type omitted).
"""

import logging
import os

from groq import AsyncGroq
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

from app.config import settings
from app.dependencies import AgentDependencies
from app.groq_compat import install_groq_tool_call_compat
from app.prompts import SYSTEM_PROMPT
from app.tools import get_course_by_name, search_knowledge

logger = logging.getLogger(__name__)


def create_agent() -> Agent:
    """
    Construct and return a fully configured PydanticAI Agent.

    output_type is intentionally omitted (defaults to str).
    """
    os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
    install_groq_tool_call_compat()

    model_name = settings.model_name
    if model_name.startswith("groq:"):
        groq_client = AsyncGroq(
            api_key=settings.groq_api_key,
            max_retries=settings.groq_max_retries,
            timeout=settings.groq_request_timeout_s,
        )
        groq_model = GroqModel(
            model_name.removeprefix("groq:"),
            provider=GroqProvider(groq_client=groq_client),
            settings={"parallel_tool_calls": False},
        )
    else:
        groq_model = model_name

    agent: Agent[AgentDependencies, str] = Agent(
        model=groq_model,
        system_prompt=SYSTEM_PROMPT,
        deps_type=AgentDependencies,
        tools=[search_knowledge, get_course_by_name],
        retries={"tools": settings.agent_tool_retries, "output": 1},
    )

    if not hasattr(agent, "_function_tools"):
        agent._function_tools = agent._function_toolset.tools  # type: ignore[attr-defined]

    logger.info(
        "Agent ready (groq_max_retries=%s, tool_calls_limit=%s, llm_turns=%s)",
        settings.groq_max_retries,
        settings.agent_tool_calls_limit,
        settings.session_llm_turns,
    )
    return agent
