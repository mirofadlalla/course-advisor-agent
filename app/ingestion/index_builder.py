from llama_index.core import VectorStoreIndex

from app.ingestion.loader import MarkdownLoader
from app.ingestion.parser import DocumentParser
from app.embeddings.provider import EmbeddingProvider


class IndexBuilder:

    def __init__(self):

        self.loader = MarkdownLoader()
        self.parser = DocumentParser()
        self.embed_model = EmbeddingProvider.get()

    def build(self):

        documents = self.loader.load()

        nodes = self.parser.parse(documents)

        index = VectorStoreIndex(

            nodes,

            embed_model=self.embed_model

        )

        return index