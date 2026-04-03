from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()


class GeminiEmbeddingModel:

    MODEL = "gemini-embedding-2-preview"

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        return self._client

    def get_text_embedding(self, text: str) -> list[float]:
        client = self._get_client()
        result = client.models.embed_content(
            model=self.MODEL,
            contents=[
                types.Content(parts=[types.Part(text=text)])
            ],
        )
        return result.embeddings[0].values

    def get_image_embedding(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> list[float]:
        client = self._get_client()
        result = client.models.embed_content(
            model=self.MODEL,
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                    ]
                )
            ],
        )
        return result.embeddings[0].values

    def get_multimodal_embedding(self, text: str, image_bytes: bytes, mime_type: str = "image/jpeg") -> list[float]:
        """Returns one aggregated embedding for combined text + image input."""
        client = self._get_client()
        result = client.models.embed_content(
            model=self.MODEL,
            contents=[
                types.Content(
                    parts=[
                        types.Part(text=text),
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    ]
                )
            ],
        )
        return result.embeddings[0].values
