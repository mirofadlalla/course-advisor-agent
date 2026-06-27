"""Usage log repository."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from app.database.mongo import MongoDatabase
from app.schemas.usage_log import UsageLog


class IUsageLogRepository(ABC):
    @abstractmethod
    async def insert(self, log: UsageLog) -> UsageLog: ...

    @abstractmethod
    async def list_logs(
        self,
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[UsageLog]: ...

    @abstractmethod
    async def aggregate_costs(self, match: dict[str, Any] | None = None) -> dict[str, Any]: ...

    @abstractmethod
    async def count(self) -> int: ...


class UsageLogRepository(IUsageLogRepository):
    def __init__(self, db: MongoDatabase) -> None:
        self._col = db.collection("usage_logs")

    async def insert(self, log: UsageLog) -> UsageLog:
        doc = log.model_dump(mode="json")
        await self._col.insert_one(doc)
        return log

    def _build_query(
        self,
        *,
        user_id: str | None,
        conversation_id: str | None,
        provider: str | None,
        model: str | None,
        start: datetime | None,
        end: datetime | None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {}
        if user_id:
            query["user_id"] = user_id
        if conversation_id:
            query["conversation_id"] = conversation_id
        if provider:
            query["provider"] = provider
        if model:
            query["model"] = model
        if start or end:
            ts: dict[str, Any] = {}
            if start:
                ts["$gte"] = start.isoformat()
            if end:
                ts["$lte"] = end.isoformat()
            query["timestamp"] = ts
        return query

    async def list_logs(
        self,
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[UsageLog]:
        query = self._build_query(
            user_id=user_id,
            conversation_id=conversation_id,
            provider=provider,
            model=model,
            start=start,
            end=end,
        )
        docs = await self._col.find(query, skip=skip, limit=limit, sort=[("timestamp", -1)])
        result: list[UsageLog] = []
        for doc in docs:
            doc.pop("_id", None)
            result.append(UsageLog.model_validate(doc))
        return result

    async def aggregate_costs(self, match: dict[str, Any] | None = None) -> dict[str, Any]:
        pipeline: list[dict[str, Any]] = []
        if match:
            pipeline.append({"$match": match})
        pipeline.extend(
            [
                {
                    "$group": {
                        "_id": None,
                        "total_cost": {"$sum": "$total_cost"},
                        "prompt_cost": {"$sum": "$prompt_cost"},
                        "completion_cost": {"$sum": "$completion_cost"},
                        "embedding_cost": {"$sum": "$embedding_cost"},
                        "total_tokens": {"$sum": "$total_tokens"},
                        "count": {"$sum": 1},
                        "avg_latency": {"$avg": "$latency_ms"},
                    }
                }
            ]
        )
        rows = await self._col.aggregate(pipeline)
        if not rows:
            return {
                "total_cost": 0.0,
                "prompt_cost": 0.0,
                "completion_cost": 0.0,
                "embedding_cost": 0.0,
                "total_tokens": 0,
                "count": 0,
                "avg_latency": 0.0,
            }
        row = rows[0]
        return {
            "total_cost": round(row.get("total_cost", 0), 8),
            "prompt_cost": round(row.get("prompt_cost", 0), 8),
            "completion_cost": round(row.get("completion_cost", 0), 8),
            "embedding_cost": round(row.get("embedding_cost", 0), 8),
            "total_tokens": row.get("total_tokens", 0),
            "count": row.get("count", 0),
            "avg_latency": round(row.get("avg_latency", 0), 2),
        }

    async def count(self) -> int:
        return await self._col.count_documents({})


def create_usage_log_repository(db: MongoDatabase) -> IUsageLogRepository:
    return UsageLogRepository(db)
