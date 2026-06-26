from app.agent import create_agent

from app.schemas.agent import AgentResponse
# from app.tools.course_tools import search_course


from app.tools.x import search_course
class ChatService:
    def __init__(self):
        self.agent = create_agent()
        self.agent.tool_plain(search_course)


    def chat(self, question: str):
        result = self.agent.run_sync(
            question
        )
        # return AgentResponse(
        #     response=result.output
        # )

        return result.output
    
# في PydanticAI فيه نوعين من الـ Tools
# النوع الأول: Tool بدون Context

# لو الـ Tool مش محتاجة Context، عرفها كده:

# def search_course(query: str):
#     ...

# وسجلها:

# agent.tool_plain(search_course)

# وليس:

# agent.tool(search_course)