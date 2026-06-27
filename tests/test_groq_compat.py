"""Tests for Groq malformed tool-call recovery."""

import pytest

from app.groq_compat import install_groq_tool_call_compat, parse_xml_tool_call


class TestParseXmlToolCall:
    @pytest.mark.parametrize(
        "failed_generation,expected_name,expected_args",
        [
            (
                '<function=get_course_by_name {"course_name": "HTML"} </function>',
                "get_course_by_name",
                {"course_name": "HTML"},
            ),
            (
                '<function=get_course_by_name({"course_name": "HTML"})</function>',
                "get_course_by_name",
                {"course_name": "HTML"},
            ),
            (
                '<function=search_knowledge {"query": "cybersecurity courses"} </function>',
                "search_knowledge",
                {"query": "cybersecurity courses"},
            ),
        ],
    )
    def test_parses_known_formats(self, failed_generation, expected_name, expected_args):
        parsed = parse_xml_tool_call(failed_generation)
        assert parsed is not None
        name, args = parsed
        assert name == expected_name
        assert args == expected_args

    def test_returns_none_for_plain_text(self):
        assert parse_xml_tool_call("maybe") is None

    def test_returns_none_for_invalid_json(self):
        assert parse_xml_tool_call("<function=foo {not json}</function>") is None


class TestIsToolUseFailed:
    def test_detects_message_without_status_code(self):
        from app.services.chat_service import _is_tool_use_failed

        exc = Exception(
            "Failed to call a function. Please adjust your prompt. "
            "See 'failed_generation' for more details."
        )
        assert _is_tool_use_failed(exc) is True


class TestInstallGroqToolCallCompat:
    def test_patch_recoveres_xml_failed_generation(self):
        install_groq_tool_call_compat()

        import pydantic_ai.models.groq as groq_module

        body = {
            "error": {
                "message": "Failed to call a function.",
                "type": "invalid_request_error",
                "code": "tool_use_failed",
                "failed_generation": (
                    '<function=get_course_by_name {"course_name": "HTML"} </function>'
                ),
            }
        }
        result = groq_module._parse_tool_use_failed_error(body)
        assert result.name == "get_course_by_name"
        assert result.arguments == {"course_name": "HTML"}
