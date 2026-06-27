"""Persist usage logs for every model call."""

from __future__ import annotations

import logging

from app.cost.calculator import CostBreakdown, calculate_embedding_cost, calculate_llm_cost, merge_costs
from app.repositories.usage_log_repository import IUsageLogRepository
from app.schemas.usage_log import UsageLog

logger = logging.getLogger(__name__)


class UsageService:
    """Record token usage and costs."""

    def __init__(self, usage_repo: IUsageLogRepository) -> None:
        self._repo = usage_repo

    async def log_model_call(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message_id: str,
        model_key: str,
        prompt_tokens: int,
        completion_tokens: int,
        embedding_tokens: int = 0,
        embedding_model_key: str = "BAAI/bge-m3",
        latency_ms: float,
    ) -> UsageLog:
        llm_cost = calculate_llm_cost(model_key, prompt_tokens, completion_tokens)
        embedding_cost = (
            calculate_embedding_cost(embedding_model_key, embedding_tokens)
            if embedding_tokens
            else CostBreakdown()
        )
        merged = merge_costs(llm_cost, embedding_cost)

        log = UsageLog(
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            provider=merged.provider,
            model=merged.model,
            model_type=merged.model_type,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens + embedding_tokens,
            embedding_tokens=embedding_tokens,
            prompt_cost=merged.prompt_cost,
            completion_cost=merged.completion_cost,
            embedding_cost=merged.embedding_cost,
            total_cost=merged.total_cost,
            latency_ms=latency_ms,
        )
        saved = await self._repo.insert(log)
        logger.debug(
            "Usage log: user=%s conv=%s cost=$%.8f",
            user_id,
            conversation_id,
            saved.total_cost,
        )
        return saved
