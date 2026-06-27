"""Week 3 compliance integration tests (mocked LLM / CRM)."""

from unittest.mock import MagicMock

import pytest

from app.prompts import SYSTEM_PROMPT
from app.repositories.crm_repository import InMemoryCrmRepository
from app.sales.intent import VisitorIntent, detect_intent
from app.sales.lead_detector import detect_lead_signals
from app.services.chat_service import ChatService
from app.services.lead_service import LeadService
from tests.test_session_memory import make_deps


class TestWeek3Compliance:
    def test_system_prompt_has_sales_rag_safety(self):
        assert "Kayfa" in SYSTEM_PROMPT
        assert "search_knowledge" in SYSTEM_PROMPT or "RETRIEVAL" in SYSTEM_PROMPT
        assert "info@kayfa.io" in SYSTEM_PROMPT
        assert "PROMPT INJECTION" in SYSTEM_PROMPT

    def test_intent_taxonomy_complete(self):
        assert len(VisitorIntent) == 5

    def test_lead_to_ticket_flow(self):
        message = "I want to enroll in Fullstack. Email me at buyer@test.com"
        intent = detect_intent(message)
        signals = detect_lead_signals(message)
        assert intent == VisitorIntent.READY_TO_ENROLL
        assert signals.is_qualified

    @pytest.mark.asyncio
    async def test_end_to_end_chat_creates_ticket_when_qualified(self):
        repo = InMemoryCrmRepository()
        service = ChatService(lead_service=LeadService(repo))

        mock_result = MagicMock()
        mock_result.output = "Here is how to enroll."
        service.agent = MagicMock()
        service.agent.run_sync.return_value = mock_result

        result = service.chat(
            "I want to register. My email is hot@lead.com",
            make_deps(),
            session_id="compliance-session",
        )

        assert result["lead_qualified"] is True
        assert result["ticket_id"] is not None
        assert result["visitor_intent"] == VisitorIntent.READY_TO_ENROLL.value

    def test_kb_coverage_module_exists(self):
        from tests import test_kb_coverage

        assert hasattr(test_kb_coverage, "EXPECTED_MARKDOWN_FILES")
