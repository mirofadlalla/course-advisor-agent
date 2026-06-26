from app.repositories.course_repository import CourseRepository
from app.schemas.CourseSearchResult_ import CourseSearchResult

repo = CourseRepository()

def search_course(
    name: str,
) -> CourseSearchResult:

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