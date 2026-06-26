
from app.config import settings

class VectorStoreFactory:

    def create():

        if settings.vector_store == "qdrant":
            pass

        elif settings.vector_store == "chroma":
            pass

        elif settings.vector_store == "simple":
            pass