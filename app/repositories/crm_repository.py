"""CRM ticket persistence — MongoDB with in-memory fallback."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime

from app.schemas.crm_ticket import CrmTicket

logger = logging.getLogger(__name__)


class ICrmRepository(ABC):
    @abstractmethod
    async def insert_ticket(self, ticket: CrmTicket) -> CrmTicket:
        """Persist a ticket; raise DuplicateTicketError if duplicate."""

    @abstractmethod
    async def get_ticket(self, ticket_id: str) -> CrmTicket | None:
        """Retrieve a ticket by id."""


class DuplicateTicketError(Exception):
    """Raised when a duplicate lead ticket would be created."""


class InMemoryCrmRepository(ICrmRepository):
    """Fallback store when MongoDB is not configured."""

    def __init__(self) -> None:
        self._tickets: dict[str, CrmTicket] = {}
        self._duplicate_keys: set[str] = set()

    async def insert_ticket(self, ticket: CrmTicket) -> CrmTicket:
        key = ticket.duplicate_key()
        if key and key in self._duplicate_keys:
            raise DuplicateTicketError(f"Duplicate ticket for key: {key}")

        ticket.updated_at = datetime.now(UTC)
        self._tickets[ticket.ticket_id] = ticket
        if key:
            self._duplicate_keys.add(key)
        return ticket

    async def get_ticket(self, ticket_id: str) -> CrmTicket | None:
        return self._tickets.get(ticket_id)


class MongoCrmRepository(ICrmRepository):
    """MongoDB-backed CRM repository using Motor."""

    def __init__(self, uri: str, database: str, collection: str) -> None:
        from motor.motor_asyncio import AsyncIOMotorClient

        self._client = AsyncIOMotorClient(uri)
        self._collection = self._client[database][collection]
        logger.info("MongoCrmRepository connected to %s.%s", database, collection)

    async def insert_ticket(self, ticket: CrmTicket) -> CrmTicket:
        key = ticket.duplicate_key()
        if key:
            existing = await self._collection.find_one({"duplicate_key": key})
            if existing:
                raise DuplicateTicketError(f"Duplicate ticket for key: {key}")

        doc = ticket.model_dump(mode="json")
        doc["duplicate_key"] = key
        doc["created_at"] = ticket.created_at
        doc["updated_at"] = datetime.now(UTC)

        try:
            await self._collection.insert_one(doc)
        except Exception as exc:
            logger.error("MongoDB insert failed: %s", exc)
            raise

        return ticket

    async def get_ticket(self, ticket_id: str) -> CrmTicket | None:
        doc = await self._collection.find_one({"ticket_id": ticket_id})
        if not doc:
            return None
        doc.pop("_id", None)
        doc.pop("duplicate_key", None)
        return CrmTicket.model_validate(doc)


def create_crm_repository(
    mongodb_uri: str,
    mongodb_database: str,
    mongodb_collection: str,
) -> ICrmRepository:
    """Factory: MongoDB when URI is set, otherwise in-memory."""
    if mongodb_uri.strip():
        return MongoCrmRepository(mongodb_uri, mongodb_database, mongodb_collection)
    logger.warning("MONGODB_URI not set — using InMemoryCrmRepository.")
    return InMemoryCrmRepository()
