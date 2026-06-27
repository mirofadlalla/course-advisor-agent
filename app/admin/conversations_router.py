"""Conversation API for authenticated users."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.auth.dependencies import get_current_user_payload
from app.schemas.conversation import ConversationCreate, ConversationPublic
from app.schemas.message import MessagePublic
from app.schemas.user import TokenPayload

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationPublic)
async def create_conversation(
    request: Request,
    payload: Annotated[TokenPayload, Depends(get_current_user_payload)],
    body: ConversationCreate | None = None,
):
    service = request.app.state.conversation_service
    conv = await service.create_conversation(payload.sub, body)
    return ConversationPublic(
        conversation_id=conv.conversation_id,
        user_id=conv.user_id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.get("", response_model=list[ConversationPublic])
async def list_conversations(
    request: Request,
    payload: Annotated[TokenPayload, Depends(get_current_user_payload)],
    skip: int = 0,
    limit: int = 50,
):
    service = request.app.state.conversation_service
    convs = await service.list_conversations(payload.sub, skip=skip, limit=limit)
    return [
        ConversationPublic(
            conversation_id=c.conversation_id,
            user_id=c.user_id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in convs
    ]


@router.get("/{conversation_id}/messages", response_model=list[MessagePublic])
async def get_messages(
    conversation_id: str,
    request: Request,
    payload: Annotated[TokenPayload, Depends(get_current_user_payload)],
):
    service = request.app.state.conversation_service
    conv = await service.get_conversation(conversation_id, payload.sub)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return await service.get_history(conversation_id)
