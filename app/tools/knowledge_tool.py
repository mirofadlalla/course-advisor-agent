"""
tools/knowledge_tool.py — Primary RAG Search Tool

RESPONSIBILITY: Expose the knowledge base to the PydanticAI agent as a Tool.

HOW PYDANTIC AI TOOLS WORK:
    1. The LLM decides it needs information to answer a question
    2. It calls the tool with structured arguments (generated from the docstring + type hints)
    3. The tool runs, calls the knowledge repository, and returns data
    4. The LLM receives the data and synthesizes a natural language answer

WHY A TOOL (not a context injection)?
    Option A: Inject all knowledge into the system prompt.
        - Problems: Context window limits (can't fit 50k tokens of docs)
        - Expensive: every API call sends all the docs
        - Static: same context regardless of what the user asks

    Option B: Retrieval Tool (this approach).
        - Dynamic: only retrieve what's relevant to the current question
        - Efficient: agent sends only the relevant chunks to the LLM
        - Scalable: works for 100k+ documents (only top-4 chunks used)

WHY RETURN list[SearchResult] INSTEAD OF A STRING?
    Option A: Return a formatted string.
        return f"Course: Python\nPrice: $40\nDuration: 10 hours"
        Problems:
        - Brittle: string format changes = bugs
        - LLM may confuse the structure with the answer
        - No metadata: can't log which source was used

    Option B: Return structured data (list[SearchResult]).
        The LLM receives structured data and synthesizes the answer itself.
        Benefits:
        - Traceable: you know which chunks were used
        - Flexible: LLM decides how to format the answer
        - Type-safe: Pydantic validates the return type

TOOL DOCSTRING:
    THIS IS CRITICAL. PydanticAI passes the docstring to the LLM as the
    tool description. The LLM reads this to decide WHEN and HOW to call the tool.
    A bad docstring = the LLM doesn't call the tool when it should,
    or calls it with wrong arguments.

    Write the docstring as if you're instructing the LLM directly.
"""

import logging

from pydantic_ai import RunContext

from app.dependencies import AgentDependencies
from app.schemas.search import SearchResult

logger = logging.getLogger(__name__)


async def search_knowledge(
    ctx: RunContext[AgentDependencies],
    query: str,
    doc_type: str | None = None,
) -> list[SearchResult]:
    """
    Search the Kayfa knowledge base for information relevant to the user's question.

    Use this tool for ALL questions about Kayfa's offerings, company, and policies.
    Always call this tool BEFORE answering. Do not answer from memory.

    Use this tool when the user asks about:
    - Specific courses: name, duration, level, prerequisites, instructor, price, link
    - Learning tracks and roadmaps: what's included, skills gained, total duration
    - Diplomas and bootcamps: curriculum, duration, career outcomes, enrollment
    - Pricing: course prices, track prices, subscription options
    - Company information: who is Kayfa, contact details, office locations
    - Policies: refunds, cancellations, payment methods, subscriptions
    - FAQs: access, deadlines, certificates, previews
    - Instructors: credentials, specializations
    - Free content: what's available for free

    Parameters:
        query:    The specific question or topic to search for.
                  Be specific: "Python course duration" is better than "Python".
        doc_type: Optional filter to narrow results to a specific category.
                  Options: "course", "roadmap", "markdown"
                  Leave as None to search across all content types.

    Returns:
        A list of relevant text chunks from the knowledge base.
        Each result includes: text content, source file, document type, relevance score.
        Synthesize these results into a clear, helpful answer for the user.
    """
    repo = ctx.deps.knowledge_repository
    settings = ctx.deps.settings

    # Build filters from optional doc_type parameter
    filters = {"doc_type": doc_type} if doc_type else None

    logger.info(
        f"search_knowledge: query='{query[:60]}', "
        f"doc_type={doc_type}, "
        f"top_k={settings.rerank_top_k}"
    )

    results = await repo.search(
        query=query,
        top_k=settings.rerank_top_k,
        filters=filters,
    )

    if not results:
        logger.warning(f"search_knowledge: No results found for query='{query}'")

    logger.info(f"search_knowledge: Returning {len(results)} results to agent.")
    return results
