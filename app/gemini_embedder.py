import os

from google import genai
from google.genai import types


def _guess_mime_type(filename: str | None) -> str:
    if not filename:
        return "image/png"

    filename_lower = filename.lower()
    if filename_lower.endswith(".jpg") or filename_lower.endswith(".jpeg"):
        return "image/jpeg"
    if filename_lower.endswith(".webp"):
        return "image/webp"
    return "image/png"


class GeminiEmbeddingModel:
    def __init__(self, model_name: str | None = None):
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")

        self.model_name = model_name or os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")
        self.client = genai.Client(api_key=api_key)

    def _embed_content(self, content: types.Content) -> list[float]:
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=[content],
        )

        if not result.embeddings:
            raise RuntimeError("Gemini embedding response did not include embeddings")

        return [float(x) for x in result.embeddings[0].values]

    def get_text_embedding(self, text: str) -> list[float]:
        content = types.Content(parts=[types.Part(text=text)])
        return self._embed_content(content)

    def get_image_embedding(self, image_bytes: bytes, mime_type: str = "image/png") -> list[float]:
        content = types.Content(
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ]
        )
        return self._embed_content(content)

    def get_image_and_text_embedding(
        self,
        text: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> list[float]:
        content = types.Content(
            parts=[
                types.Part(text=text),
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ]
        )
        return self._embed_content(content)


def infer_mime_type(filename: str | None) -> str:
    return _guess_mime_type(filename)
