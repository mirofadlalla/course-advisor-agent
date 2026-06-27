"""Response trace repository."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from app.database.mongo import MongoDatabase
from app.schemas.trace import ResponseTrace


class ITraceRepository(ABC):
    @abstractmethod
    async def insert(self, trace: ResponseTrace) -> ResponseTrace: ...

    @abstractmethod
    async def get(self, trace_id: str) -> ResponseTrace | None: ...

    @abstractmethod
    async def list_traces(
        self,
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> list[ResponseTrace]: ...

    @abstractmethod
    async def count(self) -> int: ...


class TraceRepository(ITraceRepository):
    def __init__(self, db: MongoDatabase) -> None:
        self._col = db.collection("response_traces")

    async def insert(self, trace: ResponseTrace) -> ResponseTrace:
        doc = trace.model_dump(mode="json")
        await self._col.insert_one(doc)
        return trace

    async def get(self, trace_id: str) -> ResponseTrace | None:
        doc = await self._col.find_one({"id": trace_id})
        if not doc:
            return None
        doc.pop("_id", None)
        return ResponseTrace.model_validate(doc)

    async def list_traces(
        self,
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> list[ResponseTrace]:
        query: dict[str, Any] = {}
        if user_id:
            query["user_id"] = user_id
        if conversation_id:
            query["conversation_id"] = conversation_id
        if start or end:
            ts: dict[str, Any] = {}
            if start:
                ts["$gte"] = start.isoformat()
            if end:
                ts["$lte"] = end.isoformat()
            query["timestamp"] = ts
        docs = await self._col.find(query, skip=skip, limit=limit, sort=[("timestamp", -1)])
        result: list[ResponseTrace] = []
        for doc in docs:
            doc.pop("_id", None)
            result.append(ResponseTrace.model_validate(doc))
        return result

    async def count(self) -> int:
        return await self._col.count_documents({})


def create_trace_repository(db: MongoDatabase) -> ITraceRepository:
    return TraceRepository(db)
