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
        from app.observability.langfuse_client import observe_span

        client = self._get_client()
        with observe_span(
            "embedding",
            as_type="embedding",
            input={"text_preview": (text or "")[:500], "chars": len(text or "")},
            metadata={"agent": "embedding", "model": self.MODEL},
            tags=["embedding"],
        ) as span:
            result = client.models.embed_content(
                model=self.MODEL,
                contents=[
                    types.Content(parts=[types.Part(text=text)])
                ],
            )
            try:
                from app.admin.usage_recorder import record
                meta = getattr(result, "usage_metadata", None)
                inp = getattr(meta, "prompt_token_count", 0) or len(text.split()) * 2
                record(agent="embedding", model=self.MODEL, provider="gemini", input_tokens=int(inp or 0))
            except Exception:
                pass
            values = result.embeddings[0].values
            try:
                span.update(output={"dims": len(values) if values is not None else 0})
            except Exception:
                pass
            return values

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
