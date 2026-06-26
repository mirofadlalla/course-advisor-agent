"""
tools/course_tools.py — Course Fast-Path Tool

RESPONSIBILITY: Fast lookup of course details by name using fuzzy matching.

WHY TWO SEPARATE TOOLS (knowledge_tool + course_tools)?

    search_knowledge (RAG tool):
    - Semantic: "courses about cybersecurity for beginners"
    - Handles open-ended, semantic questions
    - Returns relevant document chunks
    - Slower (embedding + retrieval + reranking)

    get_course_by_name (this tool):
    - Exact/fuzzy name lookup: "tell me about Python Basics course"
    - Handles precise course name queries
    - Returns structured Course object with all fields
    - Faster (dict lookup with fuzzy matching)

    The agent picks the right tool based on the query type.
    Both tools get their dependencies from ctx.deps (DI — no global state).

DEPENDENCY INJECTION IN TOOLS:
    The old pattern: repo = CourseRepository() at module level.
    Problem: every tool creates its own repo, reloads the JSON file.
    This is wasteful and breaks DI.

    The new pattern: ctx.deps.course_repository — the repo is created ONCE
    in lifespan() and injected into every tool call via RunContext.
    This is the correct use of PydanticAI's dependency injection.

TOOL DOCSTRING:
    Written for the LLM to understand when to call this tool vs search_knowledge.
"""

import logging

from pydantic_ai import RunContext

from app.dependencies import AgentDependencies
from app.schemas.CourseSearchResult_ import CourseSearchResult

logger = logging.getLogger(__name__)


async def get_course_by_name(
    ctx: RunContext[AgentDependencies],
    course_name: str,
) -> CourseSearchResult:
    """
    Retrieve detailed information about a specific Kayfa course by its name.

    Use this tool when the user mentions a SPECIFIC course name, for example:
    - "Tell me about the Python Basics course"
    - "What are the details of the SQL course?"
    - "How long is the Machine Learning course?"
    - "What's the price of Data Science Diploma?"

    This tool uses fuzzy name matching, so approximate names work:
    - "Python" → finds "Python Basics"
    - "ML course" → finds "Machine Learning"

    For general questions about topics (not specific course names),
    use search_knowledge instead.

    Parameters:
        course_name: The course name (exact or approximate).

    Returns:
        Course details including: name, summary, level, duration, prerequisites, link.
        If the course is not found, returns found=False.
    """
    repo = ctx.deps.course_repository

    logger.info(f"get_course_by_name: searching for '{course_name}'")

    course = repo.find_best_course(course_name)

    if course is None:
        logger.info(f"get_course_by_name: course '{course_name}' not found.")
        return CourseSearchResult(found=False)

    logger.info(f"get_course_by_name: found '{course.name}'.")

    return CourseSearchResult(
        found=True,
        name=course.name,
        summary=course.summary,
        level=course.level,
        duration=course.duration,
        link=course.link,
    )