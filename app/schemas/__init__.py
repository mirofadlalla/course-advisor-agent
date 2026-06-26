"""
schemas/__init__.py — Schemas package public API.

Centralises all domain types so callers import from app.schemas, not
from individual submodules. This decouples callers from file structure.
"""

from app.schemas.agent import AgentResponse
from app.schemas.api import ChatRequest, ChatResponse
from app.schemas.course import Course
from app.schemas.roadmap import Roadmap
from app.schemas.search import SearchResult
from app.schemas.CourseSearchResult_ import CourseSearchResult

__all__ = [
    "AgentResponse",
    "ChatRequest",
    "ChatResponse",
    "Course",
    "Roadmap",
    "SearchResult",
    "CourseSearchResult",
]
