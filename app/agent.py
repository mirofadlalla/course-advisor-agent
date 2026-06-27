"""
agent.py — PydanticAI Agent Factory

See module history in git for the Groq parallel-tool-call fix (output_type omitted).
"""

import logging
import os

from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

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
        groq_model = GroqModel(
            model_name.removeprefix("groq:"),
            settings={"parallel_tool_calls": False},
        )
    else:
        groq_model = model_name

    agent: Agent[AgentDependencies, str] = Agent(
        model=groq_model,
        system_prompt=SYSTEM_PROMPT,
        deps_type=AgentDependencies,
        tools=[search_knowledge, get_course_by_name],
        retries={"tools": 3, "output": 1},
    )

    if not hasattr(agent, "_function_tools"):
        agent._function_tools = agent._function_toolset.tools  # type: ignore[attr-defined]

    return agent
