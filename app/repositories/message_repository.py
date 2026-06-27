"""Message repository."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.database.mongo import MongoDatabase
from app.schemas.message import Message, MessageRole


class IMessageRepository(ABC):
    @abstractmethod
    async def insert(self, message: Message) -> Message: ...

    @abstractmethod
    async def list_by_conversation(
        self, conversation_id: str, *, skip: int = 0, limit: int = 200
    ) -> list[Message]: ...

    @abstractmethod
    async def count(self, *, conversation_id: str | None = None) -> int: ...


class MessageRepository(IMessageRepository):
    def __init__(self, db: MongoDatabase) -> None:
        self._col = db.collection("messages")

    async def insert(self, message: Message) -> Message:
        doc = message.model_dump(mode="json")
        await self._col.insert_one(doc)
        return message

    async def list_by_conversation(
        self, conversation_id: str, *, skip: int = 0, limit: int = 200
    ) -> list[Message]:
        docs = await self._col.find(
            {"conversation_id": conversation_id},
            skip=skip,
            limit=limit,
            sort=[("timestamp", 1)],
        )
        result: list[Message] = []
        for doc in docs:
            doc.pop("_id", None)
            result.append(Message.model_validate(doc))
        return result

    async def count(self, *, conversation_id: str | None = None) -> int:
        query = {"conversation_id": conversation_id} if conversation_id else {}
        return await self._col.count_documents(query)


def create_message_repository(db: MongoDatabase) -> IMessageRepository:
    return MessageRepository(db)
