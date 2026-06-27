"""Tests for CRM ticket schema, Arabic builder, and repository."""

import pytest

from app.repositories.crm_repository import (
    DuplicateTicketError,
    InMemoryCrmRepository,
)
from app.sales.intent import VisitorIntent
from app.sales.lead_detector import detect_lead_signals
from app.schemas.crm_ticket import CrmTicket
from app.services.lead_service import LeadService
from app.services.ticket_builder import build_arabic_summary, build_ticket_from_signals


class TestCrmTicket:
    def test_arabic_summary_preserves_english_products(self):
        ticket = CrmTicket(
            products_interested=["Python", "Fullstack Diploma"],
            visitor_intent=VisitorIntent.READY_TO_ENROLL.value,
            buying_signals=["asked_enrollment"],
            sales_action="follow_up_enrollment",
        )
        summary = build_arabic_summary(ticket)
        assert "Python" in summary
        assert "Fullstack Diploma" in summary
        assert "عميل محتمل" in summary

    @pytest.mark.asyncio
    async def test_in_memory_insert_and_retrieve(self):
        repo = InMemoryCrmRepository()
        ticket = CrmTicket(name="Ali", email="ali@test.com", session_id="s1")
        saved = await repo.insert_ticket(ticket)
        loaded = await repo.get_ticket(saved.ticket_id)
        assert loaded is not None
        assert loaded.email == "ali@test.com"

    @pytest.mark.asyncio
    async def test_duplicate_prevention(self):
        repo = InMemoryCrmRepository()
        ticket = CrmTicket(
            session_id="sess-1",
            email="dup@test.com",
            products_interested=["Python"],
        )
        await repo.insert_ticket(ticket)
        duplicate = CrmTicket(
            session_id="sess-1",
            email="dup@test.com",
            products_interested=["Python"],
        )
        with pytest.raises(DuplicateTicketError):
            await repo.insert_ticket(duplicate)

    @pytest.mark.asyncio
    async def test_lead_service_creates_ticket(self):
        repo = InMemoryCrmRepository()
        service = LeadService(repo)
        signals = detect_lead_signals(
            "I want to enroll. Email: lead@example.com. How do I pay?"
        )
        ticket = await service.maybe_create_ticket(
            session_id="session-abc",
            message="I want to enroll. Email: lead@example.com.",
            history=[],
            assistant_reply="Great, here are the steps.",
            intent=VisitorIntent.READY_TO_ENROLL,
            signals=signals,
        )
        assert ticket is not None
        assert ticket.arabic_summary
        assert ticket.email == "lead@example.com"

    def test_build_ticket_from_signals(self):
        signals = detect_lead_signals("I want the Python course, beginner level")
        ticket = build_ticket_from_signals(
            signals,
            session_id="s2",
            visitor_intent=VisitorIntent.BROWSING,
            conversation_excerpt="hello",
        )
        assert ticket.session_id == "s2"
        assert ticket.arabic_summary
