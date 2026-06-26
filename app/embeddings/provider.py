from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.config import settings

class EmbeddingProvider:

    @staticmethod
    def get():

        return HuggingFaceEmbedding(
            model_name=settings.embedding_model
        )