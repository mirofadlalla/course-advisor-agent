from app.agent import create_agent

from app.schemas.agent import AgentResponse

class ChatService:
    def __init__(self):
        self.agent = create_agent()

    def chat(self, question: str):
        result = self.agent.run_sync(
            question
        )
        # return AgentResponse(
        #     response=result.output
        # )

        return result.output