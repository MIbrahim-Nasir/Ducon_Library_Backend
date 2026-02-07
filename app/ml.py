from sentence_transformers import SentenceTransformer
from PIL import Image

# For text embeddings only
class TextEmbeddingModel:

    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)

    def get_embedding(self, text):
        embeddings = self.model.encode(text).tolist()

        return embeddings

# For multimodal embeddings
class MultimodalEmbeddingModel:

    def __init__(self, model="clip-ViT-L-14"):
        self.model = SentenceTransformer(model)

    def get_text_embedding(self, text: str):
        text_embeddings = self.model.encode(text).tolist()

        return text_embeddings

    def get_image_embedding(self, image: Image):
        image_embeddings = self.model.encode(image).tolist()

        return image_embeddings