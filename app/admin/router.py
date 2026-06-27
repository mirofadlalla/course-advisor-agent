"""Admin analytics and management API."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request

from app.auth.dependencies import require_admin
from app.schemas.user import TokenPayload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@router.get("/stats")
async def admin_stats(
    request: Request,
    _: Annotated[TokenPayload, Depends(require_admin)],
):
    """Dashboard aggregate statistics."""
    db = request.app.state.mongo_db
    user_repo = request.app.state.user_repository
    conv_service = request.app.state.conversation_service
    usage_repo = request.app.state.usage_log_repository
    trace_repo = request.app.state.trace_repository
    crm_repo = request.app.state.crm_repository

    usage_agg = await usage_repo.aggregate_costs()
    total_users = await user_repo.count()
    total_conversations = await conv_service.count_conversations()
    total_messages = await conv_service.count_messages()
    total_traces = await trace_repo.count()
    total_usage_logs = await usage_repo.count()

    tickets_count = 0
    if hasattr(crm_repo, "_tickets"):
        tickets_count = len(crm_repo._tickets)
    elif hasattr(crm_repo, "_collection"):
        tickets_count = await crm_repo._collection.count_documents({})

    top_users = await _top_by_field(request, "user_id", limit=5)
    top_conversations = await _top_by_field(request, "conversation_id", limit=5)
    daily_cost = await _daily_cost(request)
    provider_dist = await _distribution(request, "provider")
    model_dist = await _distribution(request, "model")

    count = usage_agg.get("count") or 1
    return {
        "total_users": total_users,
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "total_leads": tickets_count,
        "total_traces": total_traces,
        "total_usage_logs": total_usage_logs,
        "total_cost": usage_agg["total_cost"],
        "average_cost": round(usage_agg["total_cost"] / count, 8),
        "average_latency_ms": usage_agg["avg_latency"],
        "embedding_cost": usage_agg["embedding_cost"],
        "llm_cost": usage_agg["prompt_cost"] + usage_agg["completion_cost"],
        "top_expensive_users": top_users,
        "top_expensive_conversations": top_conversations,
        "daily_cost": daily_cost,
        "provider_distribution": provider_dist,
        "model_distribution": model_dist,
        "optimization": await _optimization_metrics(request),
    }


@router.get("/users")
async def admin_users(
    request: Request,
    _: Annotated[TokenPayload, Depends(require_admin)],
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    users = await request.app.state.user_repository.list_users(skip=skip, limit=limit)
    return {
        "users": [u.to_public().model_dump(mode="json") for u in users],
        "skip": skip,
        "limit": limit,
    }


@router.get("/conversations")
async def admin_conversations(
    request: Request,
    _: Annotated[TokenPayload, Depends(require_admin)],
    user_id: str | None = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    repo = request.app.state.conversation_repository
    if user_id:
        convs = await repo.list_by_user(user_id, skip=skip, limit=limit)
    else:
        col = repo._col
        docs = await col.find({}, skip=skip, limit=limit, sort=[("updated_at", -1)])
        from app.schemas.conversation import Conversation

        convs = []
        for doc in docs:
            doc.pop("_id", None)
            convs.append(Conversation.model_validate(doc))
    return {
        "conversations": [c.model_dump(mode="json") for c in convs],
        "skip": skip,
        "limit": limit,
    }


@router.get("/costs")
async def admin_costs(
    request: Request,
    _: Annotated[TokenPayload, Depends(require_admin)],
    user_id: str | None = None,
    conversation_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    start: str | None = None,
    end: str | None = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    logs = await request.app.state.usage_log_repository.list_logs(
        user_id=user_id,
        conversation_id=conversation_id,
        provider=provider,
        model=model,
        start=_parse_dt(start),
        end=_parse_dt(end),
        skip=skip,
        limit=limit,
    )
    agg = await request.app.state.usage_log_repository.aggregate_costs(
        {
            k: v
            for k, v in {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "provider": provider,
                "model": model,
            }.items()
            if v
        }
        or None
    )
    return {
        "logs": [log.model_dump(mode="json") for log in logs],
        "aggregate": agg,
        "skip": skip,
        "limit": limit,
    }


@router.get("/traces")
async def admin_traces(
    request: Request,
    _: Annotated[TokenPayload, Depends(require_admin)],
    user_id: str | None = None,
    conversation_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    traces = await request.app.state.trace_repository.list_traces(
        user_id=user_id,
        conversation_id=conversation_id,
        start=_parse_dt(start),
        end=_parse_dt(end),
        skip=skip,
        limit=limit,
    )
    return {
        "traces": [
            {**t.model_dump(mode="json"), "replay_steps": t.to_replay_steps()}
            for t in traces
        ],
        "skip": skip,
        "limit": limit,
    }


@router.get("/traces/{trace_id}")
async def admin_trace_detail(
    trace_id: str,
    request: Request,
    _: Annotated[TokenPayload, Depends(require_admin)],
):
    trace = await request.app.state.trace_repository.get(trace_id)
    if not trace:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Trace not found")
    return {
        **trace.model_dump(mode="json"),
        "replay_steps": trace.to_replay_steps(),
    }


@router.get("/tickets")
async def admin_tickets(
    request: Request,
    _: Annotated[TokenPayload, Depends(require_admin)],
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    crm_repo = request.app.state.crm_repository
    tickets: list[Any] = []
    if hasattr(crm_repo, "_tickets"):
        all_tickets = list(crm_repo._tickets.values())
        tickets = all_tickets[skip : skip + limit]
        tickets = [t.model_dump(mode="json") for t in tickets]
    elif hasattr(crm_repo, "_collection"):
        docs = await crm_repo._collection.find(
            {}, skip=skip, limit=limit, sort=[("created_at", -1)]
        )
        for doc in docs:
            doc.pop("_id", None)
            doc.pop("duplicate_key", None)
            tickets.append(doc)
    return {"tickets": tickets, "skip": skip, "limit": limit}


async def _top_by_field(request: Request, field: str, limit: int = 5) -> list[dict]:
    col = request.app.state.usage_log_repository._col
    pipeline = [
        {
            "$group": {
                "_id": f"${field}",
                "total_cost": {"$sum": "$total_cost"},
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"total_cost": -1}},
        {"$limit": limit},
    ]
    rows = await col.aggregate(pipeline)
    return [{"id": r["_id"], "total_cost": r.get("total_cost", 0), "count": r.get("count", 0)} for r in rows]


async def _daily_cost(request: Request) -> list[dict]:
    col = request.app.state.usage_log_repository._col
    logs = await col.find({}, limit=1000, sort=[("timestamp", -1)])
    buckets: dict[str, float] = {}
    for log in logs:
        ts = str(log.get("timestamp", ""))[:10]
        buckets[ts] = buckets.get(ts, 0) + log.get("total_cost", 0)
    return [{"date": d, "cost": round(c, 8)} for d, c in sorted(buckets.items())]


async def _distribution(request: Request, field: str) -> list[dict]:
    col = request.app.state.usage_log_repository._col
    pipeline = [
        {"$group": {"_id": f"${field}", "count": {"$sum": 1}, "total_cost": {"$sum": "$total_cost"}}},
        {"$sort": {"count": -1}},
    ]
    rows = await col.aggregate(pipeline)
    return [
        {"name": r["_id"], "count": r.get("count", 0), "total_cost": r.get("total_cost", 0)}
        for r in rows
        if r.get("_id")
    ]


async def _optimization_metrics(request: Request) -> dict[str, Any]:
    trace_repo = request.app.state.trace_repository
    usage_repo = request.app.state.usage_log_repository
    traces = await trace_repo.list_traces(limit=500)
    usage_agg = await usage_repo.aggregate_costs()

    if not traces:
        return {
            "avg_tool_calls": 0,
            "avg_retrieved_chunks": 0,
            "avg_tokens": 0,
            "avg_latency_ms": usage_agg["avg_latency"],
            "avg_cost": 0,
            "tool_usage_frequency": {},
            "retrieval_frequency": 0,
        }

    tool_counts: dict[str, int] = {}
    retrieval_hits = 0
    total_tools = 0
    total_chunks = 0
    expensive_prompts = sorted(traces, key=lambda t: t.total_cost, reverse=True)[:5]

    for trace in traces:
        total_tools += len(trace.tool_calls)
        total_chunks += len(trace.retrieved_chunks)
        if trace.retrieved_chunks:
            retrieval_hits += 1
        for tc in trace.tool_calls:
            tool_counts[tc.tool_name] = tool_counts.get(tc.tool_name, 0) + 1

    n = len(traces)
    count = usage_agg.get("count") or 1
    return {
        "avg_tool_calls": round(total_tools / n, 2),
        "avg_retrieved_chunks": round(total_chunks / n, 2),
        "avg_tokens": round(usage_agg["total_tokens"] / count, 2),
        "avg_latency_ms": usage_agg["avg_latency"],
        "avg_cost": round(usage_agg["total_cost"] / count, 8),
        "most_expensive_prompts": [
            {"prompt": t.user_prompt[:80], "cost": t.total_cost} for t in expensive_prompts
        ],
        "tool_usage_frequency": tool_counts,
        "retrieval_frequency": round(retrieval_hits / n * 100, 1),
    }
