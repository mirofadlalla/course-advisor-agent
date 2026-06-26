



# لاحظ إنه ميعرفش أى حاجة عن LlamaIndex أو Chroma.
class KnowledgeRepository:

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def search(
        self,
        query: str,
    ):
        return self.vector_store.search(query)