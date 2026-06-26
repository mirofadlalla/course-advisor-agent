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

def search_course(
    ctx: RunContext[AgentDependencies],
    name: str,

) -> CourseSearchResult:

    repo = ctx.deps.course_repository
    course = repo.find_course_by_name(name)

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