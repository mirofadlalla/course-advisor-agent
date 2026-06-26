from app.repositories.course_repository import CourseRepository
from app.schemas.CourseSearchResult_ import CourseSearchResult
from pydantic_ai import RunContext
from app.dependencies import AgentDependencies

repo = CourseRepository()
# ليه أنا ضد
# repo = CourseRepository()
# ؟
# لأنها بتكسر Dependency Injection.
# كل Tool هتعمل
# CourseRepository()
# بنفسها.
# الصح

# Application Startup
# ↓
# Create Repository
# ↓
# Create Mongo
# ↓
# Create Chroma
# ↓
# Create Logger
# ↓
# Create Dependencies
# ↓
# Agent
# ↓

# Tool يأخدهم من
# ctx.deps
# وده غلط.

def get_course_by_name(
    ctx: RunContext[AgentDependencies],
    course_name: str,

) -> CourseSearchResult:
    """
    Retrieve detailed information about a Kayfa course by its exact or partial name.

    Use this tool whenever the user asks about:
    - a specific course
    - course details
    - duration
    - level
    - course link
    """
    repo = ctx.deps.course_repository
    course = repo.find_best_course(course_name)

    if course is None:
        return CourseSearchResult(
            found=False
        )

    return CourseSearchResult(
        found=True,
        name=course.name,
        summary=course.summary,
        level=course.level,
        duration=course.duration,
        link=course.link,
    )

# ليه الـ Tool مش بترد String؟

# غلط جدًا تعمل:
# return f"""
# Python
# 20 hours
# ...
# """

# الأصح:
# ترجع Data.
# والـ LLM هو اللى يصيغ الرد.
# وده بيدى مرونة أكبر.