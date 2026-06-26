from fastapi import FastAPI

from app.config import settings

# from app.agent import agent

from app.services.chat_service import ChatService   
from contextlib import asynccontextmanager
from app.repositories.course_repository import CourseRepository
from app.repositories.roadmap_repository import RoadmapRepository

from app.dependencies import AgentDependencies
from fastapi import Request

from app.repositories.knowledge_repository import KnowledgeRepository
from app.vectorstores.vector_store_factory import VectorStoreFactory

@asynccontextmanager
async def lifespan(app: FastAPI):

    course_repo = CourseRepository()

    roadmap_repo = RoadmapRepository()

    vector_store = VectorStoreFactory.create()

    knowledge_repo = KnowledgeRepository(
        vector_store=vector_store
    )

    deps = AgentDependencies(

        course_repository=course_repo,

        roadmap_repository=roadmap_repo,

        knowledge_repository=knowledge_repo
    )

    app.state.agent_dependencies = deps
    app.state.chat_service = ChatService()

    yield

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan
    )

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
def chat(chat_request: ChatRequest, request: Request):
    return request.app.state.chat_service.chat(chat_request.message,
                               deps = request.app.state.agent_dependencies)



# course_repo = CourseRepository()
# @app.get('/courses/{course_name}')
# def get_course(course_name: str):
#     return course_repo.find_course_by_name(course_name)

# roadmap_repo = RoadmapRepository()
# @app.get('/roadmaps/{roadmap_name}')
# def get_roadmap(roadmap_name: str):
#     return roadmap_repo.find_roadmap_by_name(roadmap_name)