from app.agent import create_agent

from app.schemas.agent import AgentResponse
# from app.tools.course_tools import search_course
from app.dependencies import AgentDependencies

from app.tools.x import get_course_by_name
class ChatService:
    def __init__(self):
        self.agent = create_agent()
        self.agent.tool(get_course_by_name, sequential=True)
        print("Tool Registered")


    def chat(self, question: str, deps: AgentDependencies):
        result = self.agent.run_sync(
            question,
            deps=deps
        )

        print(result)
        print(result.all_messages())

        if isinstance(result.output, AgentResponse):
            return {"response": result.output.response}

        return {"response": str(result.output)}
    
# في PydanticAI فيه نوعين من الـ Tools
# النوع الأول: Tool بدون Context

# لو الـ Tool مش محتاجة Context، عرفها كده:

# def search_course(query: str):
#     ...

# وسجلها:

# agent.tool_plain(search_course)

# وليس:

# agent.tool(search_course)