"""
schemas/agent.py — Agent Output Schema

WHY WE REMOVED output_type=AgentResponse FROM THE AGENT:
─────────────────────────────────────────────────────────
When you set `output_type=SomePydanticModel` in PydanticAI, the framework
does NOT simply parse the model's text response into that schema.

Instead, it registers the model as a HIDDEN TOOL called `final_result`.
The LLM sees this tool in its tool list alongside your real tools.

The LLM's tool list becomes:
    • search_knowledge       ← your tool
    • get_course_by_name     ← your tool
    • final_result           ← injected by PydanticAI for structured output

WHAT GOES WRONG WITH GROQ (llama-3.3-70b-versatile):
─────────────────────────────────────────────────────
Groq's llama-3.3-70b supports PARALLEL tool calling. When given a question
like "Tell me about the MS SQL course", it generates ONE response that calls
MULTIPLE tools simultaneously:

    Turn 1 (model response):
        tool_call_1: final_result(response="To provide info, I will look up...")
        tool_call_2: get_course_by_name(course_name="MS SQL Server Programming")

PydanticAI processes tool calls in order. It sees `final_result` was called
and treats that as the terminal event — the agent loop exits immediately.

`get_course_by_name` DID execute (the log shows it returned correctly),
but its result was never fed back to the model because the loop had
already terminated when `final_result` was processed.

The execution trace from the logs:
    ✗ ACTUAL:   User → Model → [final_result + get_course_by_name] → EXIT
    ✓ EXPECTED: User → Model → get_course_by_name → Tool result → Model → final_result → EXIT

THE FIX:
────────
Remove `output_type` from the Agent entirely. PydanticAI's default output
type is `str` — the model returns its final text response naturally, and the
agent loop only terminates when the model stops calling tools and returns
plain text. This eliminates the `final_result` tool from the LLM's tool list,
so parallel-tool-call interference is impossible.

The AgentResponse class is kept here for future use (e.g., streaming,
post-processing, adding source citations as a second field). It is no longer
passed to Agent(output_type=...).
"""

from pydantic import BaseModel


class AgentResponse(BaseModel):
    """
    Structured response container.

    NOTE: Not currently used as Agent output_type.
    See module docstring for why output_type was removed.
    Kept for future use: citations, confidence scores, structured metadata.
    """

    response: str
