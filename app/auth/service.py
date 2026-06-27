"""Authentication service."""

from __future__ import annotations

import logging

from app.auth.jwt_handler import JWTError, create_access_token, create_refresh_token, decode_token
from app.auth.password import hash_password, verify_password
from app.config import Settings
from app.repositories.user_repository import IUserRepository, UserAlreadyExistsError
from app.schemas.user import TokenPair, User, UserCreate, UserLogin, UserPublic, UserRole

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """Base authentication error."""


class InvalidCredentialsError(AuthError):
    """Invalid email or password."""


class AuthService:
    """Register, login, and token refresh."""

    def __init__(self, user_repo: IUserRepository, settings: Settings) -> None:
        self._users = user_repo
        self._settings = settings

    async def register(
        self, data: UserCreate, *, allow_admin: bool = False
    ) -> tuple[UserPublic, TokenPair]:
        role = data.role
        if role == UserRole.ADMIN and not allow_admin:
            user_count = await self._users.count()
            if user_count > 0:
                role = UserRole.USER
        user = User(
            full_name=data.full_name,
            email=str(data.email).lower(),
            password_hash=hash_password(data.password),
            role=role,
        )
        try:
            saved = await self._users.create(user)
        except UserAlreadyExistsError as exc:
            raise AuthError(str(exc)) from exc
        tokens = self._issue_tokens(saved)
        logger.info("Registered user %s", saved.email)
        return saved.to_public(), tokens

    async def login(self, data: UserLogin) -> tuple[UserPublic, TokenPair]:
        user = await self._users.get_by_email(data.email.lower())
        if not user or not verify_password(data.password, user.password_hash):
            raise InvalidCredentialsError("Invalid email or password")
        tokens = self._issue_tokens(user)
        return user.to_public(), tokens

    async def refresh(self, refresh_token: str) -> TokenPair:
        try:
            payload = decode_token(
                self._settings, refresh_token, expected_type="refresh"
            )
        except JWTError as exc:
            raise AuthError(str(exc)) from exc
        user = await self._users.get_by_id(payload.sub)
        if not user:
            raise AuthError("User not found")
        return self._issue_tokens(user)

    async def get_user(self, user_id: str) -> UserPublic | None:
        user = await self._users.get_by_id(user_id)
        return user.to_public() if user else None

    def _issue_tokens(self, user: User) -> TokenPair:
        return TokenPair(
            access_token=create_access_token(
                self._settings,
                user_id=user.id,
                email=user.email,
                role=user.role,
            ),
            refresh_token=create_refresh_token(
                self._settings,
                user_id=user.id,
                email=user.email,
                role=user.role,
            ),
        )
