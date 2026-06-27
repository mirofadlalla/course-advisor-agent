"""
Centralized provider pricing configuration.

All cost calculations must read from this module — never hardcode rates in services.
Prices are USD per 1,000,000 tokens unless noted otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelType(StrEnum):
    LLM = "llm"
    EMBEDDING = "embedding"
    RERANKER = "reranker"


@dataclass(frozen=True)
class ModelPricing:
    """Pricing for a single model."""

    provider: str
    model: str
    model_type: ModelType
    input_per_1m: float = 0.0
    output_per_1m: float = 0.0
    embedding_per_1m: float = 0.0


# ─── Provider registry ────────────────────────────────────────────────────────

PRICING_REGISTRY: dict[str, ModelPricing] = {
    "groq:llama-3.3-70b-versatile": ModelPricing(
        provider="Groq",
        model="llama-3.3-70b-versatile",
        model_type=ModelType.LLM,
        input_per_1m=0.59,
        output_per_1m=0.79,
    ),
    "groq:llama-3.1-8b-instant": ModelPricing(
        provider="Groq",
        model="llama-3.1-8b-instant",
        model_type=ModelType.LLM,
        input_per_1m=0.05,
        output_per_1m=0.08,
    ),
    "openai:gpt-4o-mini": ModelPricing(
        provider="OpenAI",
        model="gpt-4o-mini",
        model_type=ModelType.LLM,
        input_per_1m=0.15,
        output_per_1m=0.60,
    ),
    "BAAI/bge-m3": ModelPricing(
        provider="BGE",
        model="bge-m3",
        model_type=ModelType.EMBEDDING,
        embedding_per_1m=0.0,  # local — no API cost
    ),
    "bge-reranker": ModelPricing(
        provider="BGE",
        model="bge-reranker",
        model_type=ModelType.RERANKER,
        embedding_per_1m=0.0,
    ),
    "cohere:rerank": ModelPricing(
        provider="Cohere",
        model="rerank-english-v3.0",
        model_type=ModelType.RERANKER,
        input_per_1m=1.0,
        output_per_1m=0.0,
    ),
    "voyage:voyage-3": ModelPricing(
        provider="Voyage",
        model="voyage-3",
        model_type=ModelType.EMBEDDING,
        embedding_per_1m=0.06,
    ),
}


def get_pricing(model_key: str) -> ModelPricing:
    """Resolve pricing for a model key; falls back to Groq default LLM pricing."""
    if model_key in PRICING_REGISTRY:
        return PRICING_REGISTRY[model_key]
    normalized = model_key.removeprefix("groq:")
    for key, pricing in PRICING_REGISTRY.items():
        if pricing.model == normalized or key.endswith(normalized):
            return pricing
    return PRICING_REGISTRY["groq:llama-3.3-70b-versatile"]
