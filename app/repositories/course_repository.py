import json
from pathlib import Path
from typing import Optional

from app.schemas.course import Course
from app.schemas.roadmap import Roadmap


class CourseRepository:

    def __init__(self):

        self.courses = self._load_courses()

        self.roadmaps = self._load_roadmaps()

    def _load_courses(self):

        path = Path("data/json/kayfa_courses.json")

        with open(path, encoding="utf-8") as f:

            data = json.load(f)

            # Convert Into List Of Objects Instead of List Of Dicts
            return [
                Course.model_validate(course)
                for course in data
            ]

    def _load_roadmaps(self):

        path = Path("data/json/kayfa_roadmaps.json")

        with open(path, encoding="utf-8") as f:

            data = json.load(f)

            return [
                Roadmap.model_validate(roadmap)
                for roadmap in data
            ]
    
    def find_course_by_name(
        self,
        name: str
    ) -> Optional[Course]:

        name = name.lower().strip()

        for course in self.courses:

            course_name = course.name.lower()

            if name in course_name:
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