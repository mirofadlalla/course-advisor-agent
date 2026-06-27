"""Tests for backend conversation memory via session_id."""

from unittest.mock import MagicMock

import pytest

from app.services.chat_service import ChatService
from app.services.lead_service import LeadService
from app.services.session_store import SessionStore


def make_deps():
    from app.config import Settings
    from app.dependencies import AgentDependencies

    return AgentDependencies(
        course_repository=MagicMock(),
        roadmap_repository=MagicMock(),
        knowledge_repository=MagicMock(),
        settings=Settings(groq_api_key="test-key"),
    )


class TestSessionMemory:
    def test_session_store_appends_and_retrieves_history(self):
        store = SessionStore(max_messages=10)
        store.append_turn("sess-1", "Hello", "Hi there")
        store.append_turn("sess-1", "Second question", "Second answer")

        history = store.get_history("sess-1")
        assert len(history) == 4  # 2 user + 2 assistant ModelMessage objects
        assert store.get_user_messages("sess-1") == [
            "Hello",
            "Second question",
        ]

    def test_chat_passes_message_history_to_agent(self):
        store = SessionStore()
        store.append_turn("sess-x", "First message", "First reply")

        service = ChatService.__new__(ChatService)
        service.session_store = store
        service.lead_service = LeadService(
            __import__(
                "app.repositories.crm_repository",
                fromlist=["InMemoryCrmRepository"],
            ).InMemoryCrmRepository()
        )

        mock_result = MagicMock()
        mock_result.output = "Answer two"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result
        service.agent = mock_agent

        service.chat("Second message", make_deps(), session_id="sess-x")

        call_kwargs = mock_agent.run_sync.call_args.kwargs
        assert call_kwargs.get("message_history") is not None
        assert len(call_kwargs["message_history"]) == 2
        assert call_kwargs.get("instructions")

    @pytest.mark.asyncio
    async def test_finalize_turn_stores_new_messages(self):
        store = SessionStore()
        service = ChatService(session_store=store)
        meta = await service._finalize_turn(
            session_id="s-new",
            question="Q1",
            response="A1",
            user_history=[],
            intent_value="browsing",
        )
        assert meta["visitor_intent"] == "browsing"
        assert store.get_user_messages("s-new") == ["Q1"]
