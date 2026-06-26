import json
from pathlib import Path
from typing import Optional

from app.schemas.course import Course
from app.schemas.roadmap import Roadmap


class RoadmapRepository:

    def __init__(self):

        self.roadmaps = self._load_roadmaps()


    def _load_roadmaps(self):

        path = Path("data/json/kayfa_roadmaps.json")

        with open(path, encoding="utf-8") as f:

            data = json.load(f)

            return [
                Roadmap.model_validate(roadmap)
                for roadmap in data
            ]
    
    def find_roadmap_by_name(
        self,
        name: str
    ) -> Optional[Roadmap]:

        name = name.lower().strip()

        for roadmap in self.roadmaps:

            roadmap_name = roadmap.name.lower()

            if name in roadmap_name:
                return roadmap

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