"""Conversation and message persistence service."""

from __future__ import annotations

import logging

from app.repositories.conversation_repository import IConversationRepository
from app.repositories.message_repository import IMessageRepository
from app.schemas.conversation import Conversation, ConversationCreate
from app.schemas.message import Message, MessagePublic, MessageRole

logger = logging.getLogger(__name__)


class ConversationService:
    """Manage conversations and persisted message history."""

    def __init__(
        self,
        conversation_repo: IConversationRepository,
        message_repo: IMessageRepository,
    ) -> None:
        self._conversations = conversation_repo
        self._messages = message_repo

    async def create_conversation(
        self, user_id: str, data: ConversationCreate | None = None
    ) -> Conversation:
        data = data or ConversationCreate()
        conv = Conversation(user_id=user_id, title=data.title)
        return await self._conversations.create(conv)

    async def get_conversation(
        self, conversation_id: str, user_id: str
    ) -> Conversation | None:
        conv = await self._conversations.get(conversation_id)
        if conv and conv.user_id == user_id:
            return conv
        return None

    async def list_conversations(
        self, user_id: str, *, skip: int = 0, limit: int = 50
    ) -> list[Conversation]:
        return await self._conversations.list_by_user(user_id, skip=skip, limit=limit)

    async def save_user_message(
        self, conversation_id: str, content: str
    ) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=content,
        )
        await self._messages.insert(msg)
        await self._conversations.touch(conversation_id)
        title = content[:60].strip() or "New conversation"
        conv = await self._conversations.get(conversation_id)
        if conv and conv.title == "New conversation":
            await self._conversations.update_title(conversation_id, title)
        return msg

    async def save_assistant_message(
        self,
        conversation_id: str,
        content: str,
        *,
        tokens: int = 0,
        cost: float = 0.0,
    ) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=content,
            tokens=tokens,
            cost=cost,
        )
        await self._messages.insert(msg)
        await self._conversations.touch(conversation_id)
        return msg

    async def get_history(
        self, conversation_id: str, *, limit: int = 200
    ) -> list[MessagePublic]:
        messages = await self._messages.list_by_conversation(
            conversation_id, limit=limit
        )
        return [
            MessagePublic(
                id=m.id,
                conversation_id=m.conversation_id,
                role=m.role,
                content=m.content,
                timestamp=m.timestamp,
                tokens=m.tokens,
                cost=m.cost,
            )
            for m in messages
        ]

    async def count_messages(self) -> int:
        return await self._messages.count()

    async def count_conversations(self) -> int:
        return await self._conversations.count()
