from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)

    def get_embedding(self, text):
        embeddings = self.model.encode(text).tolist()

        return embeddings
