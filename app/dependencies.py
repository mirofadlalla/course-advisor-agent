"""
dependencies.py — Agent Dependencies Container

RESPONSIBILITY: Define the AgentDependencies dataclass — the single object
passed to every PydanticAI tool via RunContext[AgentDependencies].

WHY @dataclass AND NOT BaseModel?
    BaseModel is for DATA validation (API payloads, DB records).
    AgentDependencies holds SERVICES (repositories, settings) — objects with
    behaviour, not data. @dataclass is the right container: no validation
    overhead, no JSON serialisation, just a typed named-tuple for DI.

DEPENDENCY INVERSION:
    knowledge_repository is typed as IKnowledgeRepository (the abstract
    interface), NOT KnowledgeRepository (the concrete class).
    This means:
    - Tools never import KnowledgeRepository directly.
    - Tests can inject MockKnowledgeRepository without touching tool code.
    - Swapping the retrieval backend requires zero changes to tool code.

WHAT LIVES HERE:
    course_repository     → fast exact/fuzzy course lookup (CourseRepository)
    roadmap_repository    → roadmap lookup (RoadmapRepository)
    knowledge_repository  → full RAG pipeline (IKnowledgeRepository interface)
    settings              → runtime config (top_k values, model names, etc.)
                            Tools read settings to avoid hardcoded thresholds.
"""

from dataclasses import dataclass

from app.config import Settings
from app.repositories.base import IKnowledgeRepository
from app.repositories.course_repository import CourseRepository
from app.repositories.roadmap_repository import RoadmapRepository


@dataclass
class AgentDependencies:
    """
    Container for all objects a PydanticAI tool can request via RunContext.

    Constructed ONCE in lifespan() and stored on app.state.
    Every request shares the same repositories and settings — no re-creation.
    """

    course_repository: CourseRepository
    """Fast-path fuzzy lookup for specific course names."""

    roadmap_repository: RoadmapRepository
    """Roadmap/learning-path lookup by name."""

    knowledge_repository: IKnowledgeRepository
    """
    Full RAG pipeline: Hybrid retrieval (Dense + BM25) + optional Reranker.
    Typed as the INTERFACE — concrete class is an implementation detail.
    """

    settings: Settings
    """
    Runtime configuration. Tools read rerank_top_k, model names, etc.
    Injected here so tools never call Settings() directly (no global state in tools).
    """
