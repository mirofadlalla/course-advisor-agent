from llama_index.core.node_parser import MarkdownNodeParser


class DocumentParser:

    def __init__(self):
        self.parser = MarkdownNodeParser()

    def parse(self, documents):
        return self.parser.get_nodes_from_documents(documents)


# Test
# # from app.ingestion.loaders import MarkdownLoader
# from loaders import MarkdownLoader
# # from app.ingestion.parsers import DocumentParser

# loader = MarkdownLoader()

# documents = loader.load()

# parser = DocumentParser()

# nodes = parser.parse(documents)

# print(len(nodes))

# print(nodes[0].text)

# print(nodes[0].metadata)