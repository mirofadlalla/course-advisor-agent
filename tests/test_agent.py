"""
tests/test_agent.py — Agent and ChatService Tests

All PydanticAI / Groq calls are mocked — no real LLM traffic.
Tests verify that:
  - create_agent() returns a configured Agent
  - ChatService.chat() delegates to the agent and returns {response: str}
  - The agent is only created once (shared across requests)
"""

from unittest.mock import MagicMock, patch

# ── Helpers ────────────────────────────────────────────────────────────────────


def make_deps():
    from app.config import Settings
    from app.dependencies import AgentDependencies

    return AgentDependencies(
        course_repository=MagicMock(),
        roadmap_repository=MagicMock(),
        knowledge_repository=MagicMock(),
        settings=Settings(groq_api_key="test-key"),
    )


# ── create_agent ───────────────────────────────────────────────────────────────


class TestCreateAgent:
    def test_create_agent_returns_agent(self):
        from pydantic_ai import Agent

        from app.agent import create_agent

        agent = create_agent()
        assert isinstance(agent, Agent)

    def test_agent_has_string_output_type(self):
        """output_type must be str (not AgentResponse) to avoid the Groq parallel-tool bug."""
        from app.agent import create_agent

        agent = create_agent()
        # PydanticAI stores the output type on _result_schema or _output_type
        # The key check: no final_result tool injected
        tool_names = [t.name for t in agent._function_tools.values()]
        assert "final_result" not in tool_names

    def test_agent_has_expected_tools(self):
        from app.agent import create_agent

        agent = create_agent()
        tool_names = [t.name for t in agent._function_tools.values()]
        assert "search_knowledge" in tool_names
        assert "get_course_by_name" in tool_names


# ── ChatService ────────────────────────────────────────────────────────────────


class TestChatService:
    def _make_service_with_mock_agent(self, agent_output: str):
        """Build a ChatService whose agent is replaced by a mock."""
        from app.services.chat_service import ChatService

        service = ChatService.__new__(ChatService)

        mock_result = MagicMock()
        mock_result.output = agent_output

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        service.agent = mock_agent
        return service

    def test_chat_returns_response_dict(self):
        service = self._make_service_with_mock_agent("Here are the courses.")
        result = service.chat("What courses are available?", make_deps())

        assert isinstance(result, dict)
        assert "response" in result
        assert result["response"] == "Here are the courses."

    def test_chat_passes_question_to_agent(self):
        service = self._make_service_with_mock_agent("answer")
        deps = make_deps()

        service.chat("Tell me about Python", deps)

        service.agent.run_sync.assert_called_once_with("Tell me about Python", deps=deps)

    def test_chat_passes_deps_to_agent(self):
        service = self._make_service_with_mock_agent("answer")
        deps = make_deps()

        service.chat("question", deps)

        call_kwargs = service.agent.run_sync.call_args
        assert call_kwargs.kwargs["deps"] is deps

    def test_chat_returns_string_response(self):
        service = self._make_service_with_mock_agent("A string response")
        result = service.chat("hi", make_deps())
        assert isinstance(result["response"], str)

    def test_chat_service_creates_agent_on_init(self):
        """ChatService.__init__ must call create_agent exactly once."""
        with patch("app.services.chat_service.create_agent") as mock_create:
            mock_create.return_value = MagicMock()
            from app.services.chat_service import ChatService

            ChatService()

            mock_create.assert_called_once()

    def test_agent_shared_across_calls(self):
        """All calls must use the same agent instance — no re-creation per request."""
        service = self._make_service_with_mock_agent("resp")
        agent_before = service.agent

        service.chat("q1", make_deps())
        service.chat("q2", make_deps())

        assert service.agent is agent_before


# ── System prompt ──────────────────────────────────────────────────────────────


class TestSystemPrompt:
    def test_system_prompt_is_non_empty_string(self):
        from app.prompts import SYSTEM_PROMPT

        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT.strip()) > 100  # meaningful content

    def test_system_prompt_contains_key_sections(self):
        from app.prompts import SYSTEM_PROMPT
        from app.prompts.behavior import BEHAVIOR
        from app.prompts.identity import IDENTITY

        # Each sub-prompt should appear in the combined prompt
        assert IDENTITY.strip()[:30] in SYSTEM_PROMPT
        assert BEHAVIOR.strip()[:30] in SYSTEM_PROMPT
