from pydantic import BaseModel


class Roadmap(BaseModel):
    id: str | None = None

    provider: str | None = None
    host: str | None = None

    name: str | None = None

    summary: str | None = None

    track: list[str] | None = None

    skills: list[str] | None = None

    tools: list[str] | None = None

    duration: str | None = None

    courses_count: int | None = None

    link: str | None = None

    courses_list: list[str] | None = None