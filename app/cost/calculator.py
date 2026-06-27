"""Cost calculation using centralized pricing configuration."""

from __future__ import annotations

from dataclasses import dataclass

from app.cost.pricing import ModelType, get_pricing


@dataclass
class CostBreakdown:
    """Token costs broken down by category."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    embedding_cost: float = 0.0
    total_cost: float = 0.0
    provider: str = ""
    model: str = ""
    model_type: str = "llm"

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
            "prompt_cost": self.prompt_cost,
            "completion_cost": self.completion_cost,
            "embedding_cost": self.embedding_cost,
            "total_cost": self.total_cost,
            "provider": self.provider,
            "model": self.model,
            "model_type": self.model_type,
        }


def calculate_llm_cost(
    model_key: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> CostBreakdown:
    """Calculate LLM input/output costs for a model call."""
    pricing = get_pricing(model_key)
    prompt_cost = (prompt_tokens / 1_000_000) * pricing.input_per_1m
    completion_cost = (completion_tokens / 1_000_000) * pricing.output_per_1m
    total = prompt_cost + completion_cost
    return CostBreakdown(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_cost=round(prompt_cost, 8),
        completion_cost=round(completion_cost, 8),
        total_cost=round(total, 8),
        provider=pricing.provider,
        model=pricing.model,
        model_type=pricing.model_type.value,
    )


def calculate_embedding_cost(model_key: str, embedding_tokens: int) -> CostBreakdown:
    """Calculate embedding cost for retrieval/reranking."""
    pricing = get_pricing(model_key)
    embedding_cost = (embedding_tokens / 1_000_000) * pricing.embedding_per_1m
    return CostBreakdown(
        embedding_tokens=embedding_tokens,
        embedding_cost=round(embedding_cost, 8),
        total_cost=round(embedding_cost, 8),
        provider=pricing.provider,
        model=pricing.model,
        model_type=ModelType.EMBEDDING.value,
    )


def merge_costs(*parts: CostBreakdown) -> CostBreakdown:
    """Merge multiple cost breakdowns into a single total."""
    merged = CostBreakdown()
    for part in parts:
        merged.prompt_tokens += part.prompt_tokens
        merged.completion_tokens += part.completion_tokens
        merged.embedding_tokens += part.embedding_tokens
        merged.prompt_cost += part.prompt_cost
        merged.completion_cost += part.completion_cost
        merged.embedding_cost += part.embedding_cost
        merged.total_cost += part.total_cost
        if not merged.provider and part.provider:
            merged.provider = part.provider
            merged.model = part.model
            merged.model_type = part.model_type
    merged.prompt_cost = round(merged.prompt_cost, 8)
    merged.completion_cost = round(merged.completion_cost, 8)
    merged.embedding_cost = round(merged.embedding_cost, 8)
    merged.total_cost = round(merged.total_cost, 8)
    return merged
