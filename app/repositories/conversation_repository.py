"""Conversation repository."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime

from app.database.mongo import MongoDatabase
from app.schemas.conversation import Conversation


class IConversationRepository(ABC):
    @abstractmethod
    async def create(self, conversation: Conversation) -> Conversation: ...

    @abstractmethod
    async def get(self, conversation_id: str) -> Conversation | None: ...

    @abstractmethod
    async def list_by_user(
        self, user_id: str, *, skip: int = 0, limit: int = 50
    ) -> list[Conversation]: ...

    @abstractmethod
    async def update_title(self, conversation_id: str, title: str) -> None: ...

    @abstractmethod
    async def touch(self, conversation_id: str) -> None: ...

    @abstractmethod
    async def count(self, *, user_id: str | None = None) -> int: ...


class ConversationRepository(IConversationRepository):
    def __init__(self, db: MongoDatabase) -> None:
        self._col = db.collection("conversations")

    async def create(self, conversation: Conversation) -> Conversation:
        doc = conversation.model_dump(mode="json")
        await self._col.insert_one(doc)
        return conversation

    async def get(self, conversation_id: str) -> Conversation | None:
        doc = await self._col.find_one({"conversation_id": conversation_id})
        if not doc:
            return None
        doc.pop("_id", None)
        return Conversation.model_validate(doc)

    async def list_by_user(
        self, user_id: str, *, skip: int = 0, limit: int = 50
    ) -> list[Conversation]:
        docs = await self._col.find(
            {"user_id": user_id},
            skip=skip,
            limit=limit,
            sort=[("updated_at", -1)],
        )
        result: list[Conversation] = []
        for doc in docs:
            doc.pop("_id", None)
            result.append(Conversation.model_validate(doc))
        return result

    async def update_title(self, conversation_id: str, title: str) -> None:
        await self._col.update_one(
            {"conversation_id": conversation_id},
            {"$set": {"title": title, "updated_at": datetime.now(UTC).isoformat()}},
        )

    async def touch(self, conversation_id: str) -> None:
        await self._col.update_one(
            {"conversation_id": conversation_id},
            {"$set": {"updated_at": datetime.now(UTC).isoformat()}},
        )

    async def count(self, *, user_id: str | None = None) -> int:
        query = {"user_id": user_id} if user_id else {}
        return await self._col.count_documents(query)


def create_conversation_repository(db: MongoDatabase) -> IConversationRepository:
    return ConversationRepository(db)
