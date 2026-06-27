import json
from pathlib import Path

from app.schemas.roadmap import Roadmap


class RoadmapRepository:
    def __init__(self):
        self.roadmaps = self._load_roadmaps()

    def _load_roadmaps(self):
        path = Path("data/json/kayfa_roadmaps.json")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

            return [Roadmap.model_validate(roadmap) for roadmap in data]

    def find_roadmap_by_name(self, name: str) -> Roadmap | None:
        name = name.lower().strip()

        for roadmap in self.roadmaps:
            roadmap_name = roadmap.name.lower()

            if name in roadmap_name:
                return roadmap

        return None
