"""
groq_compat.py — Groq tool_use_failed recovery for malformed tool-call XML.

Groq's llama-3.3-70b occasionally emits tool calls as plain text instead of
structured API tool calls, e.g.:

    <function=get_course_by_name {"course_name": "HTML"} </function>

The Groq API rejects these with HTTP 400 code="tool_use_failed" and puts the
malformed string in `failed_generation`.  PydanticAI's built-in parser only
handles JSON-shaped failed_generation, so the tool is never executed.

This module extends that parser to recover name + arguments from the XML-like
formats Groq models sometimes produce, so the agent can still run the tool.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

# Patterns for legacy Groq/Llama text tool-call formats.
# 1) <function=name {"key": "val"} </function>  (missing ">" after name)
_XML_JSON_RE = re.compile(
    r"^<function=([^\s{>(]+)\s*(\{.*?\})\s*(?:</function>)?\s*$",
    re.DOTALL,
)
# 2) <function=name({"key": "val"})</function>
_XML_PARENS_RE = re.compile(
    r"^<function=([^(<]+)\((\{.*?\})\)\s*(?:</function>)?\s*$",
    re.DOTALL,
)

_compat_installed = False


def parse_xml_tool_call(failed_generation: str) -> tuple[str, dict[str, Any]] | None:
    """
    Parse a malformed Groq failed_generation string into (tool_name, arguments).

    Returns None when the string is not a recognized XML-like tool call.
    """
    text = failed_generation.strip()
    if not text.startswith("<function="):
        return None

    for pattern in (_XML_JSON_RE, _XML_PARENS_RE):
        match = pattern.match(text)
        if not match:
            continue
        name = match.group(1).strip()
        try:
            arguments = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        if isinstance(arguments, dict):
            return name, arguments

    return None


def install_groq_tool_call_compat() -> None:
    """
    Patch pydantic-ai's Groq failed_generation parser once per process.

    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _compat_installed
    if _compat_installed:
        return

    import pydantic_ai.models.groq as groq_module

    original = groq_module._parse_tool_use_failed_error

    def _parse_tool_use_failed_error(body: Any) -> Any:
        result = original(body)
        if isinstance(result, groq_module._GroqToolUseFailedGeneration):
            return result
        if isinstance(result, str):
            parsed = parse_xml_tool_call(result)
            if parsed is not None:
                name, arguments = parsed
                try:
                    return groq_module._GroqToolUseFailedGeneration(
                        name=name,
                        arguments=arguments,
                    )
                except ValidationError:
                    pass
        return result

    groq_module._parse_tool_use_failed_error = _parse_tool_use_failed_error
    _compat_installed = True
