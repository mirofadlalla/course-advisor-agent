"""CRM ticket schema for qualified sales leads."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

ContactPreference = Literal["phone", "whatsapp", "email", "any"]


class CrmTicket(BaseModel):
    """Full CRM ticket payload stored in MongoDB."""

    ticket_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = ""
    name: str = ""
    phone: str = ""
    whatsapp: str = ""
    email: str = ""
    city: str = ""
    language: str = ""
    dialect: str = ""
    contact_preference: ContactPreference = "any"
    products_interested: list[str] = Field(default_factory=list)
    goal: str = ""
    level: str = ""
    buying_signals: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    visitor_intent: str = ""
    arabic_summary: str = ""
    sales_action: str = ""
    conversation_excerpt: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def duplicate_key(self) -> str:
        """Key used to prevent duplicate tickets for the same lead."""
        parts = [
            self.session_id,
            self.email.lower(),
            self.phone,
            self.whatsapp,
            ",".join(sorted(self.products_interested)),
        ]
        return "|".join(p for p in parts if p)
