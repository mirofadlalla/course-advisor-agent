"""FastAPI authentication dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.jwt_handler import JWTError, decode_token
from app.config import settings
from app.schemas.user import TokenPayload, UserRole

_bearer = HTTPBearer(auto_error=False)


def _get_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def get_current_user_payload(
    token: Annotated[str, Depends(_get_token)],
) -> TokenPayload:
    try:
        return decode_token(settings, token, expected_type="access")
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def require_admin(
    payload: Annotated[TokenPayload, Depends(get_current_user_payload)],
) -> TokenPayload:
    if payload.role != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return payload


def get_optional_user_payload(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> TokenPayload | None:
    if credentials is None or not credentials.credentials:
        return None
    try:
        return decode_token(settings, credentials.credentials, expected_type="access")
    except JWTError:
        return None


def get_auth_service(request: Request):
    return request.app.state.auth_service


def get_auth_service_optional(request: Request):
    return getattr(request.app.state, "auth_service", None)


def get_chat_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> TokenPayload:
    """Resolve authenticated user for chat; allows bypass when auth_disabled."""
    if settings.auth_disabled:
        return TokenPayload(
            sub="dev-user",
            email="dev@local",
            role="user",
            type="access",
        )
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login required before chatting",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_token(settings, credentials.credentials, expected_type="access")
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
