
from dataclasses import dataclass

from app.repositories.course_repository import CourseRepository
from app.repositories.roadmap_repository import RoadmapRepository


@dataclass
class AgentDependencies:

    course_repository: CourseRepository

    roadmap_repository: RoadmapRepository

'''
ليه Dataclass؟
لأنها مجرد Container.
مش محتاجة Validation.
مش API.
مش Database Model.
إحنا بس بنجمع Objects.

لأن BaseModel معمول للبيانات (Data Validation).
لكن CourseRepository مش بيانات، ده Service.
إحنا عايزين Container بسيط.
وده بالضبط دور @dataclass.
'''

