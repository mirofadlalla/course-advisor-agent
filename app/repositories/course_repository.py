import json
from pathlib import Path

from rapidfuzz import process

from app.schemas.course import Course


class CourseRepository:
    def __init__(self):
        self.courses = self._load_courses()

    def _load_courses(self):
        path = Path("data/json/kayfa_courses.json")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

            # Convert Into List Of Objects Instead of List Of Dicts
            return [Course.model_validate(course) for course in data]

    # Exact Match
    def find_by_exact_name(
        self,
        query: str,
    ):
        query = query.strip().lower()

        for course in self.courses:
            if course.name.lower() == query:
                return course

        return None

    # Contains Match
    def find_by_keyword(
        self,
        query: str,
    ):
        query = query.strip().lower()

        for course in self.courses:
            course_name = course.name.lower()

            if query in course_name:
                return course

            if course_name in query:
                return course

        return None

    # Fuzzy Match
    def find_by_fuzzy(
        self,
        query: str,
    ):
        names = [course.name for course in self.courses]

        result = process.extractOne(
            query,
            names,
        )

        if result is None:
            return None

        best_name, score, _ = result

        if score < 80:
            return None

        for course in self.courses:
            if course.name == best_name:
                return course

        return None

    # Find Best Course
    def find_best_course(
        self,
        query: str,
    ):
        course = self.find_by_exact_name(query)

        if course:
            return course

        course = self.find_by_keyword(query)

        if course:
            return course

        course = self.find_by_fuzzy(query)

        if course:
            return course

        return None


# repo = CourseRepository()

# course = repo.find_course_by_name("python")

# print(course)

# ليه Repository بيرجع dict؟
# لأن JSON أصلاً عبارة عن Dictionaries.
# لكن..
#  ده مش أفضل Design.
# التصميم الأفضل
# نعمل
# schemas/
# وفيه
# Course
# يعنى بدل
# dict
# يبقى
# Course

# مثلاً:
# class Course(BaseModel):
#     id: int
#     name: str
#     summary: str
#     level: str
#     duration: str
#     track: str
#     link: str

# وساعتها الـ Repository يرجع:
# Course
# بدل Dictionary.
# وده هيخلينا نستفيد من:

# Validation
# Autocomplete
# Type Safety
# أخطاء أقل
