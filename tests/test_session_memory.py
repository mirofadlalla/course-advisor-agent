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
        assert len(history) == 4
        assert store.get_user_messages("sess-1") == [
            "Hello",
            "Second question",
        ]

    def test_llm_history_limited_to_last_n_turns(self):
        store = SessionStore(llm_turns=2, assistant_history_chars=1000)
        for i in range(4):
            store.append_turn("sess-2", f"Q{i}", f"A{i}")

        llm_history = store.get_llm_history("sess-2")
        assert len(llm_history) == 4  # 2 turns × 2 messages
        user_texts = [
            p.content
            for msg in llm_history
            if hasattr(msg, "parts")
            for p in msg.parts
            if hasattr(p, "content") and isinstance(p.content, str) and p.content.startswith("Q")
        ]
        assert user_texts == ["Q2", "Q3"]

    def test_llm_history_truncates_long_assistant_replies(self):
        store = SessionStore(llm_turns=1, assistant_history_chars=20)
        store.append_turn("sess-3", "Q", "A" * 100)

        llm_history = store.get_llm_history("sess-3")
        assistant_text = llm_history[1].parts[0].content  # type: ignore[index, union-attr]
        assert len(assistant_text) <= 21
        assert assistant_text.endswith("…")

    def test_recent_user_messages_capped_for_analysis(self):
        store = SessionStore(analysis_user_messages=2)
        for i in range(5):
            store.append_turn("sess-4", f"msg-{i}", "ok")

        recent = store.get_recent_user_messages("sess-4")
        assert recent == ["msg-3", "msg-4"]

    def test_chat_passes_limited_message_history_to_agent(self):
        store = SessionStore(llm_turns=3)
        store.append_turn("sess-x", "First message", "First reply")

        service = ChatService.__new__(ChatService)
        service.session_store = store
        service.conversation_service = None
        service.usage_service = None
        service.trace_repository = None
        service.cancellation_manager = __import__(
            "app.services.cancellation_manager", fromlist=["CancellationManager"]
        ).CancellationManager()
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
        assert call_kwargs.get("usage_limits") is not None

    @pytest.mark.asyncio
    async def test_finalize_turn_stores_new_messages(self):
        from app.services.chat_service import TurnContext

        store = SessionStore()
        service = ChatService(session_store=store)
        turn = TurnContext(
            user_id="u1",
            conversation_id="",
            message_id="m1",
            session_key="s-new",
        )
        meta = await service._finalize_turn(
            turn=turn,
            question="Q1",
            response="A1",
            user_history=[],
            intent_value="browsing",
            instructions="",
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
            model_key="groq:llama-3.3-70b-versatile",
        )
        assert meta["visitor_intent"] == "browsing"
        assert store.get_user_messages("s-new") == ["Q1"]
