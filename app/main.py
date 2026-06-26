from fastapi import FastAPI

from app.config import settings

# from app.agent import agent

from app.services.chat_service import ChatService   

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
      )

course_service = ChatService()
# agent2 = ChatService()

@app.get('/health')
def root():
    # return {
    #     "message": "Course Advisor Agent Running"
    #     }

# مينفعش باي شكل من الاشكال فاست اي بي اي يشوف اي معلومات عن الايجنت ولو حبيت اعدل الايجنت بكره هضطر اعدل كل الاندبوينت فالحل استخدم سيرفزيز
    # result = agent.run_sync(
    #     "Introduce yourself." 
    # )

    # return  {
    #     "response": result.output
    # }

    # تبقي كده 
    
    # return course_service.chat("Introduce yourself.")
    return {
        "message": "Course Advisor Agent Running"
    }

from app.schemas.api import ChatRequest, ChatResponse


@app.post('/chat', response_model=ChatResponse)
def chat(chat_request: ChatRequest):
    return course_service.chat(chat_request.message)