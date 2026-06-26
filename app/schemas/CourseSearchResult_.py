from pydantic import BaseModel

class CourseSearchResult(BaseModel):
    found: bool
    name: str | None = None
    summary: str | None = None
    level: str | None = None
    duration: str | None = None
    link: str | None = None


# ليه Model جديد؟ 
# ل ناتج الكورس مع اننا حلونا كل كورس من ديكشنري الي اويجكت

# لأن الـ Repository عنده بيانات كتير.
# لكن الـ LLM محتاج جزء صغير فقط.

# مثلاً مش محتاج:
# id
# prerequisites
# roadmaps

# كل Token زيادة = تكلفة أعلى.