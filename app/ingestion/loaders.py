from llama_index.core import SimpleDirectoryReader


class MarkdownLoader:

    def load(self):

        reader = SimpleDirectoryReader(
            input_dir="data/text",
            required_exts=[".md"]
        )

        return reader.load_data()


# loader = MarkdownLoader()

# documents = loader.load()

# print(len(documents))

# print(documents[0].metadata)

# print(documents[0].text[:300])