"""
repositories/base.py — Abstract Knowledge Repository Interface

RESPONSIBILITY: Define the contract for the knowledge store.
The agent's search tool depends ONLY on this interface — never on LlamaIndex,
Chroma, Qdrant, or any specific retrieval implementation.

REPOSITORY PATTERN:
    The Repository Pattern hides the data source behind an interface.
    Callers (tools, services) don't know:
    - Is the data in a vector DB?
    - Is it retrieved via BM25, dense, or hybrid search?
    - Is there a reranker?
    - Which embedding model was used?

    They only know: "give me relevant knowledge for this query."

    This decoupling means you can:
    1. Swap the entire retrieval stack without touching tool code
    2. Test tools with a MockKnowledgeRepository
    3. Add caching inside the repository without changing the interface
    4. Add observability (logging, tracing) in one place

COMPARE WITH THE OLD DESIGN:
    Old knowledge_repository.py:
        class KnowledgeRepository:
            def search(self, query: str):
                return self.vector_store.search(query)

    Problems:
    - No interface (tools would import the concrete class)
    - Returns raw vector store output (not a domain object)
    - No type annotations
    - No metadata filtering support

    New design:
    - IKnowledgeRepository defines the contract
    - Tools import the abstract interface
    - Returns list[SearchResult] (domain objects)
    - Supports top_k and metadata filters
"""

from abc import ABC, abstractmethod

from app.schemas.search import SearchResult


class IKnowledgeRepository(ABC):
    """
    Abstract contract for the knowledge store.

    The agent's search tool depends on this interface.
    KnowledgeRepository implements it.
    MockKnowledgeRepository (for tests) also implements it.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 4,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """
        Search the knowledge base for relevant information.

        Args:
            query:   Natural language query from the agent.
            top_k:   Maximum number of results to return.
            filters: Optional metadata filters.
                     Example: {"doc_type": "course"} → only search courses.

        Returns:
            list[SearchResult]: Ranked, relevant chunks with metadata.
                                Returns [] if nothing relevant found.
        """
        ...
