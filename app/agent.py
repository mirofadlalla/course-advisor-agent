from pydantic_ai import Agent

from app.config import settings\

from app.prompts import SYSTEM_PROMPT

from app.schemas.agent import AgentResponse
# agent = Agent(
#     model = settings.model_name,
#     system_prompt = """
#     You are a helpful Course Advisor AI.
#     Help students choose the best course.
#     Answer clearly and professionally.
#     """
# ) 
# المشكله
# # أى ملف يعمل:
# # from app.agent import agent
# # وخلاص.
# # دى اسمها:
# # Global Singleton
# # وهى مش غلط، لكن مع الوقت هتعمل مشاكل.
# زي عاوز اغير الايجنت بكود جديد، هضطر اعدل كل الملفات اللى استوردت الايجنت القديم.
# او اعمل كذا ايجنت فى نفس الوقت، هضطر اعمل كود جديد لكل ايجنت.
# الحل:


# Factory هو Object أو Function مسئوله عن إنشاء Objects.
def create_agent() -> Agent:

    return Agent(
        model=settings.model_name,
        system_prompt=SYSTEM_PROMPT,
        output_type= AgentResponse

    )
