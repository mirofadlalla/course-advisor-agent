"""Shared MongoDB access with in-memory fallback for local dev and tests."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class InMemoryCollection:
    """Minimal async-compatible in-memory MongoDB collection substitute."""

    def __init__(self) -> None:
        self._docs: list[dict[str, Any]] = []
        self._id_counter = 0

    async def insert_one(self, document: dict[str, Any]) -> Any:
        self._id_counter += 1
        doc = dict(document)
        doc["_id"] = self._id_counter
        self._docs.append(doc)
        return type("InsertResult", (), {"inserted_id": self._id_counter})()

    async def find_one(
        self, query: dict[str, Any], sort: list[tuple[str, int]] | None = None
    ) -> dict[str, Any] | None:
        matches = [d for d in self._docs if self._matches(d, query)]
        if sort:
            key, direction = sort[0]
            matches.sort(key=lambda d: d.get(key) or "", reverse=direction < 0)
        return dict(matches[0]) if matches else None

    async def find(
        self,
        query: dict[str, Any] | None = None,
        skip: int = 0,
        limit: int = 0,
        sort: list[tuple[str, int]] | None = None,
    ) -> list[dict[str, Any]]:
        query = query or {}
        matches = [d for d in self._docs if self._matches(d, query)]
        if sort:
            key, direction = sort[0]
            matches.sort(key=lambda d: d.get(key) or "", reverse=direction < 0)
        if skip:
            matches = matches[skip:]
        if limit:
            matches = matches[:limit]
        return [dict(d) for d in matches]

    async def count_documents(self, query: dict[str, Any] | None = None) -> int:
        query = query or {}
        return sum(1 for d in self._docs if self._matches(d, query))

    async def update_one(self, query: dict[str, Any], update: dict[str, Any]) -> Any:
        for doc in self._docs:
            if self._matches(doc, query):
                if "$set" in update:
                    doc.update(update["$set"])
                return type("UpdateResult", (), {"modified_count": 1})()
        return type("UpdateResult", (), {"modified_count": 0})()

    async def create_index(self, *args: Any, **kwargs: Any) -> str:
        return "in_memory_index"

    async def aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Support basic $match and $group for admin analytics."""
        docs = list(self._docs)
        for stage in pipeline:
            if "$match" in stage:
                docs = [d for d in docs if self._matches(d, stage["$match"])]
            elif "$group" in stage:
                group = stage["$group"]
                group_id = group["_id"]
                grouped: dict[Any, dict[str, Any]] = {}
                for doc in docs:
                    if group_id is None:
                        key = None
                    elif isinstance(group_id, str) and group_id.startswith("$"):
                        key = doc.get(group_id[1:])
                    else:
                        key = group_id
                    bucket = grouped.setdefault(key, {"_id": key})
                    for field, expr in group.items():
                        if field == "_id":
                            continue
                        if isinstance(expr, dict) and "$sum" in expr:
                            src = expr["$sum"]
                            if src == 1:
                                val = 1
                            elif isinstance(src, str) and src.startswith("$"):
                                val = doc.get(src[1:], 0) or 0
                            else:
                                val = src
                            bucket[field] = bucket.get(field, 0) + val
                        elif isinstance(expr, dict) and "$avg" in expr:
                            src = expr["$avg"]
                            field_name = src[1:] if isinstance(src, str) and src.startswith("$") else src
                            bucket.setdefault(f"_vals_{field}", []).append(doc.get(field_name, 0))
                for bucket in grouped.values():
                    for key in list(bucket.keys()):
                        if key.startswith("_vals_"):
                            vals = bucket.pop(key)
                            out_key = key.replace("_vals_", "")
                            bucket[out_key] = sum(vals) / len(vals) if vals else 0
                docs = list(grouped.values())
            elif "$sort" in stage:
                key = list(stage["$sort"].keys())[0]
                direction = stage["$sort"][key]
                docs.sort(key=lambda d: d.get(key) or 0, reverse=direction < 0)
            elif "$limit" in stage:
                docs = docs[: stage["$limit"]]
        return docs

    def _matches(self, doc: dict[str, Any], query: dict[str, Any]) -> bool:
        for key, expected in query.items():
            if key == "$or":
                if not any(self._matches(doc, clause) for clause in expected):
                    return False
                continue
            if isinstance(expected, dict):
                if "$gte" in expected and doc.get(key, 0) < expected["$gte"]:
                    return False
                if "$lte" in expected and doc.get(key, 0) > expected["$lte"]:
                    return False
                if "$regex" in expected:
                    import re

                    if not re.search(expected["$regex"], str(doc.get(key, "")), re.I):
                        return False
                continue
            if doc.get(key) != expected:
                return False
        return True


class InMemoryDatabase:
    """In-memory database with named collections."""

    def __init__(self) -> None:
        self._collections: dict[str, InMemoryCollection] = {}

    def __getitem__(self, name: str) -> InMemoryCollection:
        if name not in self._collections:
            self._collections[name] = InMemoryCollection()
        return self._collections[name]


class MongoDatabase:
    """Wrapper exposing collection access for Motor or in-memory backend."""

    def __init__(self, backend: Any, database_name: str, in_memory: bool = False) -> None:
        self._backend = backend
        self.database_name = database_name
        self.in_memory = in_memory

    def collection(self, name: str) -> Any:
        if self.in_memory:
            return self._backend[name]
        return self._backend[ self.database_name][name]

    async def ensure_indexes(self) -> None:
        """Create indexes for all Week 3 collections."""
        specs: list[tuple[str, list[tuple[str, int]], bool]] = [
            ("users", [("email", 1)], True),
            ("users", [("role", 1)], False),
            ("conversations", [("user_id", 1)], False),
            ("conversations", [("conversation_id", 1)], True),
            ("messages", [("conversation_id", 1)], False),
            ("messages", [("timestamp", -1)], False),
            ("usage_logs", [("user_id", 1)], False),
            ("usage_logs", [("conversation_id", 1)], False),
            ("usage_logs", [("timestamp", -1)], False),
            ("usage_logs", [("provider", 1)], False),
            ("usage_logs", [("model", 1)], False),
            ("response_traces", [("conversation_id", 1)], False),
            ("response_traces", [("user_id", 1)], False),
            ("response_traces", [("timestamp", -1)], False),
            ("tickets", [("ticket_id", 1)], True),
        ]
        for coll_name, keys, unique in specs:
            col = self.collection(coll_name)
            kwargs: dict[str, Any] = {}
            if unique:
                kwargs["unique"] = True
            await col.create_index(keys, **kwargs)


def create_mongo_database(uri: str, database: str) -> MongoDatabase:
    """Factory: Motor when URI is set, otherwise in-memory."""
    if uri.strip():
        from motor.motor_asyncio import AsyncIOMotorClient

        client = AsyncIOMotorClient(uri)
        logger.info("MongoDatabase connected to %s", database)
        return MongoDatabase(client, database, in_memory=False)
    logger.warning("MONGODB_URI not set — using in-memory MongoDatabase.")
    return MongoDatabase(InMemoryDatabase(), database, in_memory=True)
