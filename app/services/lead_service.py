"""Orchestrate lead qualification and CRM ticket creation."""

from __future__ import annotations

import logging

from app.repositories.crm_repository import DuplicateTicketError, ICrmRepository
from app.sales.intent import VisitorIntent, detect_intent
from app.sales.lead_detector import LeadSignals, detect_lead_signals
from app.schemas.crm_ticket import CrmTicket
from app.services.ticket_builder import build_ticket_from_signals

logger = logging.getLogger(__name__)


class LeadService:
    """Detect qualified leads and persist CRM tickets."""

    def __init__(self, crm_repository: ICrmRepository) -> None:
        self._crm = crm_repository
        self._session_tickets: dict[str, str] = {}

    def analyze_turn(
        self,
        message: str,
        history: list[str],
        assistant_reply: str = "",
    ) -> tuple[VisitorIntent, LeadSignals]:
        intent = detect_intent(message, history)
        signals = detect_lead_signals(message, history, assistant_reply)
        return intent, signals

    async def maybe_create_ticket(
        self,
        *,
        session_id: str | None,
        message: str,
        history: list[str],
        assistant_reply: str,
        intent: VisitorIntent,
        signals: LeadSignals,
    ) -> CrmTicket | None:
        if not session_id or not signals.is_qualified:
            return None

        if session_id in self._session_tickets:
            logger.info("Ticket already created for session %s", session_id)
            return None

        excerpt = "\n".join([*history[-6:], message, assistant_reply[:500]])
        ticket = build_ticket_from_signals(
            signals,
            session_id=session_id,
            visitor_intent=intent,
            conversation_excerpt=excerpt,
        )

        try:
            saved = await self._crm.insert_ticket(ticket)
            self._session_tickets[session_id] = saved.ticket_id
            logger.info("CRM ticket created: %s", saved.ticket_id)
            return saved
        except DuplicateTicketError as exc:
            logger.warning("Duplicate CRM ticket skipped: %s", exc)
            return None
