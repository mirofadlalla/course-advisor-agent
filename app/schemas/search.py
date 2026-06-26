"""
schemas/search.py — SearchResult Domain Object

RESPONSIBILITY: Define the data contract between the retrieval layer
and everything above it (KnowledgeRepository → Tool → Agent).

WHY NOT USE NodeWithScore DIRECTLY?
    NodeWithScore is a LlamaIndex internal type.
    If the tool/agent receives NodeWithScore:
    1. You've exposed your infrastructure (LlamaIndex) to your business logic
    2. If you switch from LlamaIndex to Haystack tomorrow, you must change
       every tool, every repository, every test that touches these objects
    3. NodeWithScore contains many internal fields the agent doesn't need

    SearchResult is YOUR domain type. It:
    - Contains only what the agent needs
    - Is independent of LlamaIndex
    - Is serializable to JSON (Pydantic BaseModel)
    - Has clear, typed fields with documentation

    The conversion happens ONCE in KnowledgeRepository.
    Everything above the repository is isolated from the infrastructure.

FIELDS RATIONALE:
    text:           The chunk content the agent will read and synthesize
    score:          Relevance score (for logging/debugging, not shown to user)
    source_file:    Which file the chunk came from (for citations)
    doc_type:       "course"|"roadmap"|"diploma"|"markdown" (for filtering logic)
    section_header: Which section of the document (for context)
    metadata:       Raw metadata dict (for forward compatibility — new fields added later)
"""

from pydantic import BaseModel


class SearchResult(BaseModel):
    """
    Domain object returned by KnowledgeRepository.search().

    This is the boundary type between infrastructure and business logic.
    The agent's search tool returns list[SearchResult] to the LLM.
    """

    text: str
    """The actual chunk text. This is what the LLM reads to answer the question."""

    score: float = 0.0
    """Relevance score from the retriever/reranker. [0.0, 1.0] or RRF score."""

    source_file: str = ""
    """Original file this chunk came from. E.g.: 'kayfa_soc_diploma.md'"""

    doc_type: str = ""
    """Document category: 'course', 'roadmap', 'diploma', 'markdown', etc."""

    section_header: str = ""
    """Markdown section header this chunk belongs to. E.g.: '## Curriculum'"""

    metadata: dict = {}
    """Raw metadata dict for forward compatibility."""
