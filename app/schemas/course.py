from pydantic import BaseModel


class Course(BaseModel):
    id: str | None = None
    provider: str | None = None
    host: str | None = None

    name: str | None = None

    track: list[str] | None = None

    summary: str | None = None

    link: str | None = None

    duration: str | None = None

    level: str | None = None

    prerequisites: str | None = None

    roadmaps: list[str] | None = None


# ليه استخدمنا Field؟
# بدل
# roadmaps = []

# كتبنا
# Field(default_factory=list)
# سؤال مهم جدًا
# ليه؟
# لو عملت
# class Course(BaseModel):

#     roadmaps = []
# كل Objects هيشاركوا نفس الـ List.

# وده اسمه:
# Mutable Default Argument Problem

# مثال:
# course1.roadmaps.append("AI")
# course2.roadmaps
# ممكن تلاقى فيها "AI" رغم إنك معدلتهاش

# أما
# Field(default_factory=list)
# كل Object بياخد List جديدة.
# وده Correct.
