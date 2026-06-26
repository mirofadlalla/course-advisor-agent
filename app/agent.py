"""
agent.py — PydanticAI Agent Factory

THE BUG THIS FILE FIXED (and why):
────────────────────────────────────────────────────────────────────────────
PREVIOUS CODE:
    Agent(
        output_type=AgentResponse,   ← THIS WAS THE BUG
        tools=[search_knowledge, get_course_by_name],
    )

WHAT output_type ACTUALLY DOES IN PYDANTIC AI:
    Setting output_type=SomePydanticModel does NOT post-process the model's
    text into a schema. It registers a hidden tool named `final_result` that
    the LLM must call to terminate the agent loop.

    The LLM's visible tool list becomes:
        • search_knowledge
        • get_course_by_name
        • final_result          ← injected, not visible in your code

GROQ PARALLEL TOOL CALLS — THE ACTUAL FAILURE:
    Groq's llama-3.3-70b supports parallel tool calling. When the model
    receives "Tell me about the MS SQL course", it generates ONE response
    calling MULTIPLE tools at once:

        tool_call_1: final_result(response="To provide info, I will look up...")
        tool_call_2: get_course_by_name(course_name="MS SQL Server")

    PydanticAI processes tool calls sequentially. `final_result` appears
    first → agent loop terminates → `get_course_by_name` result is NEVER
    fed back to the model.

    Log evidence:
        1. Model calls `final_result`       ← loop termination triggered here
        2. Model calls `get_course_by_name` ← tool ran, but result never used
        3. Tool returns correctly
        4. Agent stops                       ← because final_result already fired

THE FIX:
    Remove output_type. PydanticAI's default output is `str`.
    Without output_type, there is NO `final_result` tool in the LLM's tool list.
    The agent loop terminates only when the model returns plain text (no tool calls).
    Correct execution is restored:

        User → Model → get_course_by_name → Tool result → Model → str response → EXIT

SECONDARY ISSUE FIXED — BEHAVIOR PROMPT:
    The old prompt said: "Answer step by step."
    This caused the model to generate a reasoning plan ("I will first look up...")
    as the content of `final_result.response` while planning a tool call.
    Fixed to: "Always call the appropriate tool FIRST. Only answer after you
    have the tool results."
"""

from pydantic_ai import Agent

from app.config import settings
from app.dependencies import AgentDependencies
from app.prompts import SYSTEM_PROMPT
from app.tools import get_course_by_name, search_knowledge


def create_agent() -> Agent:
    """
    Construct and return a fully configured PydanticAI Agent.

    output_type is intentionally omitted (defaults to str).
    See module docstring for the full explanation of why.

    Returns:
        Agent[AgentDependencies, str]: Ready-to-run agent. The str output
        is the model's final natural-language response after all tool calls
        have completed and their results have been fed back to the model.
    """
    agent: Agent[AgentDependencies, str] = Agent(
        model=settings.model_name,
        system_prompt=SYSTEM_PROMPT,
        # output_type intentionally omitted — see module docstring.
        # Using str (the default) eliminates the hidden `final_result` tool
        # that causes Groq parallel-tool-call interference.
        deps_type=AgentDependencies,
        tools=[search_knowledge, get_course_by_name],
    )

    return agent
