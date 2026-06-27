"""User repository — MongoDB with in-memory fallback."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from app.database.mongo import MongoDatabase
from app.schemas.user import User, UserRole

logger = logging.getLogger(__name__)


class UserAlreadyExistsError(Exception):
    """Raised when registering a duplicate email."""


class IUserRepository(ABC):
    @abstractmethod
    async def create(self, user: User) -> User: ...

    @abstractmethod
    async def get_by_email(self, email: str) -> User | None: ...

    @abstractmethod
    async def get_by_id(self, user_id: str) -> User | None: ...

    @abstractmethod
    async def list_users(self, *, skip: int = 0, limit: int = 50) -> list[User]: ...

    @abstractmethod
    async def count(self) -> int: ...


class UserRepository(IUserRepository):
    """MongoDB-backed user repository."""

    def __init__(self, db: MongoDatabase) -> None:
        self._col = db.collection("users")

    async def create(self, user: User) -> User:
        existing = await self.get_by_email(user.email)
        if existing:
            raise UserAlreadyExistsError(f"Email already registered: {user.email}")
        doc = user.model_dump(mode="json")
        await self._col.insert_one(doc)
        return user

    async def get_by_email(self, email: str) -> User | None:
        doc = await self._col.find_one({"email": email.lower()})
        if not doc:
            return None
        doc.pop("_id", None)
        return User.model_validate(doc)

    async def get_by_id(self, user_id: str) -> User | None:
        doc = await self._col.find_one({"id": user_id})
        if not doc:
            return None
        doc.pop("_id", None)
        return User.model_validate(doc)

    async def list_users(self, *, skip: int = 0, limit: int = 50) -> list[User]:
        docs = await self._col.find({}, skip=skip, limit=limit, sort=[("created_at", -1)])
        users: list[User] = []
        for doc in docs:
            doc.pop("_id", None)
            users.append(User.model_validate(doc))
        return users

    async def count(self) -> int:
        return await self._col.count_documents({})


def create_user_repository(db: MongoDatabase) -> IUserRepository:
    return UserRepository(db)
