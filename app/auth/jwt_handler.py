"""JWT token creation and verification."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from app.config import Settings
from app.schemas.user import TokenPayload, UserRole


class JWTError(Exception):
    """Raised when a JWT is invalid or expired."""


def create_access_token(
    settings: Settings,
    *,
    user_id: str,
    email: str,
    role: UserRole,
) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_access_expire_minutes)
    payload = {
        "sub": user_id,
        "email": email,
        "role": role.value,
        "type": "access",
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(
    settings: Settings,
    *,
    user_id: str,
    email: str,
    role: UserRole,
) -> str:
    expire = datetime.now(UTC) + timedelta(days=settings.jwt_refresh_expire_days)
    payload = {
        "sub": user_id,
        "email": email,
        "role": role.value,
        "type": "refresh",
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(settings: Settings, token: str, *, expected_type: str) -> TokenPayload:
    """Decode and validate a JWT; raises JWTError on failure."""
    try:
        data: dict[str, Any] = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.PyJWTError as exc:
        raise JWTError(str(exc)) from exc

    if data.get("type") != expected_type:
        raise JWTError(f"Expected token type '{expected_type}'")

    return TokenPayload(
        sub=data["sub"],
        email=data["email"],
        role=data["role"],
        type=data["type"],
    )
