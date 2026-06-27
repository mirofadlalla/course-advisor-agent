"""
tests/test_schemas.py — Schema and Repository Unit Tests

Tests Pydantic schemas and repository lookup logic
with mocked/in-memory data — no file system or API access.
"""

import pytest
from unittest.mock import patch, MagicMock


# ── ChatRequest / ChatResponse ─────────────────────────────────────────────────

class TestAPISchemas:

    def test_chat_request_valid(self):
        from app.schemas.api import ChatRequest

        req = ChatRequest(message="What courses are available?")
        assert req.message == "What courses are available?"

    def test_chat_request_empty_string_allowed(self):
        from app.schemas.api import ChatRequest

        req = ChatRequest(message="")
        assert req.message == ""

    def test_chat_response_valid(self):
        from app.schemas.api import ChatResponse

        resp = ChatResponse(response="Here are the courses…")
        assert resp.response == "Here are the courses…"


# ── SearchResult ───────────────────────────────────────────────────────────────

class TestSearchResult:

    def test_basic_construction(self):
        from app.schemas.search import SearchResult

        r = SearchResult(text="Python course content")
        assert r.text == "Python course content"
        assert r.score == 0.0
        assert r.source_file == ""
        assert r.doc_type == ""
        assert r.section_header == ""
        assert r.metadata == {}

    def test_full_construction(self):
        from app.schemas.search import SearchResult

        r = SearchResult(
            text="SOC analyst roadmap",
            score=0.87,
            source_file="roadmap.md",
            doc_type="roadmap",
            section_header="## Tools",
            metadata={"chunk_index": 2},
        )
        assert r.score == 0.87
        assert r.doc_type == "roadmap"
        assert r.metadata["chunk_index"] == 2

    def test_serialises_to_dict(self):
        from app.schemas.search import SearchResult

        r = SearchResult(text="test", score=0.5)
        d = r.model_dump()
        assert d["text"] == "test"
        assert d["score"] == 0.5


# ── Course schema ──────────────────────────────────────────────────────────────

class TestCourseSchema:

    def test_all_fields_optional(self):
        from app.schemas.course import Course

        c = Course()
        assert c.name is None
        assert c.track is None

    def test_construction(self):
        from app.schemas.course import Course

        c = Course(
            id="1",
            name="Python for Beginners",
            level="Beginner",
            duration="20 hours",
            track=["programming"],
            summary="Learn Python basics",
            link="https://example.com",
        )
        assert c.name == "Python for Beginners"
        assert c.level == "Beginner"
        assert c.track == ["programming"]


# ── CourseRepository (mocked file I/O) ────────────────────────────────────────

FAKE_COURSES = [
    {
        "id": "1",
        "name": "Python Programming",
        "level": "Beginner",
        "duration": "20h",
        "track": ["programming"],
        "summary": "Learn Python",
        "link": "https://example.com/python",
        "provider": "Kayfa",
        "host": "Online",
        "prerequisites": None,
        "roadmaps": None,
    },
    {
        "id": "2",
        "name": "Advanced SQL Server",
        "level": "Advanced",
        "duration": "30h",
        "track": ["data"],
        "summary": "Master SQL",
        "link": "https://example.com/sql",
        "provider": "Kayfa",
        "host": "Online",
        "prerequisites": "Basic SQL",
        "roadmaps": None,
    },
    {
        "id": "3",
        "name": "Network Security Fundamentals",
        "level": "Intermediate",
        "duration": "25h",
        "track": ["security"],
        "summary": "Learn network security",
        "link": "https://example.com/netsec",
        "provider": "Kayfa",
        "host": "Online",
        "prerequisites": None,
        "roadmaps": None,
    },
]


@pytest.fixture
def course_repo():
    """CourseRepository with mocked data — no file I/O."""
    import json
    from unittest.mock import mock_open, patch
    from app.repositories.course_repository import CourseRepository

    mock_data = json.dumps(FAKE_COURSES)
    with patch("builtins.open", mock_open(read_data=mock_data)):
        with patch("pathlib.Path.open", mock_open(read_data=mock_data)):
            repo = CourseRepository.__new__(CourseRepository)
            from app.schemas.course import Course
            repo.courses = [Course.model_validate(c) for c in FAKE_COURSES]
    return repo


class TestCourseRepository:

    def test_find_by_exact_name_hit(self, course_repo):
        course = course_repo.find_by_exact_name("Python Programming")
        assert course is not None
        assert course.name == "Python Programming"

    def test_find_by_exact_name_case_insensitive(self, course_repo):
        course = course_repo.find_by_exact_name("python programming")
        assert course is not None

    def test_find_by_exact_name_miss(self, course_repo):
        course = course_repo.find_by_exact_name("Nonexistent Course XYZ")
        assert course is None

    def test_find_by_keyword_substring(self, course_repo):
        course = course_repo.find_by_keyword("SQL")
        assert course is not None
        assert "SQL" in course.name

    def test_find_by_keyword_miss(self, course_repo):
        course = course_repo.find_by_keyword("quantum computing xyz")
        assert course is None

    def test_find_by_fuzzy_close_match(self, course_repo):
        # Slight typo
        course = course_repo.find_by_fuzzy("Python Programing")  # one 'm'
        assert course is not None
        assert course.name == "Python Programming"

    def test_find_by_fuzzy_no_match_below_threshold(self, course_repo):
        course = course_repo.find_by_fuzzy("zzzzz completely unrelated")
        assert course is None

    def test_find_best_course_exact_first(self, course_repo):
        course = course_repo.find_best_course("Advanced SQL Server")
        assert course is not None
        assert course.name == "Advanced SQL Server"

    def test_find_best_course_falls_back_to_fuzzy(self, course_repo):
        course = course_repo.find_best_course("Network Securty Fundamentals")  # typo
        assert course is not None

    def test_find_best_course_no_match(self, course_repo):
        course = course_repo.find_best_course("zzzz not a course zzzz")
        assert course is None


# ── AgentDependencies ──────────────────────────────────────────────────────────

class TestAgentDependencies:

    def test_construction(self):
        from app.dependencies import AgentDependencies
        from app.config import Settings
        from unittest.mock import MagicMock

        deps = AgentDependencies(
            course_repository=MagicMock(),
            roadmap_repository=MagicMock(),
            knowledge_repository=MagicMock(),
            settings=Settings(groq_api_key="test"),
        )
        assert deps.settings.groq_api_key == "test"
