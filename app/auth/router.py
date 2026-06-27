"""Authentication API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.auth.dependencies import get_auth_service, get_current_user_payload
from app.auth.service import AuthError, AuthService, InvalidCredentialsError
from app.schemas.user import TokenPair, TokenPayload, UserCreate, UserLogin, UserPublic

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=dict)
async def register(
    body: UserCreate,
    service: Annotated[AuthService, Depends(get_auth_service)],
):
    try:
        user, tokens = await service.register(body)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"user": user.model_dump(mode="json"), **tokens.model_dump()}


@router.post("/login", response_model=dict)
async def login(
    body: UserLogin,
    service: Annotated[AuthService, Depends(get_auth_service)],
):
    try:
        user, tokens = await service.login(body)
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        ) from exc
    return {"user": user.model_dump(mode="json"), **tokens.model_dump()}


@router.post("/refresh", response_model=TokenPair)
async def refresh_token(
    request: Request,
    service: Annotated[AuthService, Depends(get_auth_service)],
):
    body = await request.json()
    token = body.get("refresh_token", "")
    if not token:
        raise HTTPException(status_code=400, detail="refresh_token required")
    try:
        return await service.refresh(token)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.get("/me", response_model=UserPublic)
async def me(
    payload: Annotated[TokenPayload, Depends(get_current_user_payload)],
    service: Annotated[AuthService, Depends(get_auth_service)],
):
    user = await service.get_user(payload.sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
