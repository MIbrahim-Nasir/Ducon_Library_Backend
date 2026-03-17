import os
from typing import Literal

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    AutoImageProcessor,
    SiglipProcessor,
    SiglipTokenizer,
    SiglipImageProcessor,
)

EmbeddingProfile = Literal["legacy", "modern"]


def _profile_from_env() -> EmbeddingProfile:
    profile = os.getenv("EMBEDDING_PROFILE", "legacy").strip().lower()
    return "modern" if profile == "modern" else "legacy"


def get_active_embedding_profile() -> EmbeddingProfile:
    return _profile_from_env()


def get_collection_names() -> dict[str, str]:
    profile = get_active_embedding_profile()
    text_base = os.getenv("TEXT_COLLECTION_BASE", "text_store")
    image_base = os.getenv("IMAGE_COLLECTION_BASE", "image_store")
    fused_base = os.getenv("FUSED_COLLECTION_BASE", "fused_store")
    return {
        "text": f"{text_base}_{profile}",
        "image": f"{image_base}_{profile}",
        "fused": f"{fused_base}_{profile}",
    }


def _normalize_vector(vector: np.ndarray) -> list[float]:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.astype(np.float32).tolist()
    return (vector / norm).astype(np.float32).tolist()


def _as_feature_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]
    if isinstance(output, (tuple, list)) and len(output) > 0:
        return _as_feature_tensor(output[0])
    raise RuntimeError("Could not convert model output to feature tensor")


def _get_device() -> str:
    configured = os.getenv("MODEL_DEVICE", "auto").strip().lower()
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return configured


class TextEmbeddingModel:
    def __init__(self, model: str | None = None):
        profile = get_active_embedding_profile()

        if model:
            model_name = model
        elif profile == "modern":
            model_name = os.getenv("TEXT_MODEL_NAME", "intfloat/e5-base-v2")
        else:
            model_name = os.getenv("TEXT_MODEL_NAME", "all-MiniLM-L6-v2")

        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, device=_get_device())

    def _encode(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.astype(np.float32).tolist()

    def get_document_embedding(self, text: str) -> list[float]:
        if "e5" in self.model_name.lower() and not text.startswith("passage:"):
            text = f"passage: {text}"
        return self._encode(text)

    def get_query_embedding(self, text: str) -> list[float]:
        if "e5" in self.model_name.lower() and not text.startswith("query:"):
            text = f"query: {text}"
        return self._encode(text)

    def get_embedding(self, text: str) -> list[float]:
        return self.get_document_embedding(text)


class MultimodalEmbeddingModel:
    def __init__(self, model: str | None = None):
        self.profile = get_active_embedding_profile()
        self.device = _get_device()

        if model:
            self.model_name = model
        elif self.profile == "modern":
            self.model_name = os.getenv("MULTIMODAL_MODEL_NAME", "google/siglip2-so400m-patch14-384")
        else:
            self.model_name = os.getenv("MULTIMODAL_MODEL_NAME", "clip-ViT-L-14")

        model_name_lc = self.model_name.lower()
        self.use_transformers = any(token in model_name_lc for token in ("siglip", "siglip2"))

        if self.use_transformers:
            is_siglip_family = "siglip" in self.model_name.lower()
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)

            # Some transformers versions fail on AutoProcessor for SigLIP2.
            # Fallback: use separate tokenizer + image processor.
            self.tokenizer = None
            self.image_processor = None
            try:
                if is_siglip_family:
                    self.processor = SiglipProcessor.from_pretrained(self.model_name)
                else:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        use_fast=False,
                    )
            except Exception:
                self.processor = None
                if is_siglip_family:
                    self.tokenizer = SiglipTokenizer.from_pretrained(self.model_name)
                    self.image_processor = SiglipImageProcessor.from_pretrained(self.model_name)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        use_fast=False,
                    )
                    self.image_processor = AutoImageProcessor.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                    )

            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.processor = None
            self.tokenizer = None
            self.image_processor = None

    def _extract_text_features(self, encoded: dict) -> torch.Tensor:
        if hasattr(self.model, "get_text_features"):
            return _as_feature_tensor(self.model.get_text_features(**encoded))

        outputs = self.model(**encoded)
        return _as_feature_tensor(outputs)

    def _extract_image_features(self, encoded: dict) -> torch.Tensor:
        if hasattr(self.model, "get_image_features"):
            return _as_feature_tensor(self.model.get_image_features(**encoded))

        outputs = self.model(**encoded)
        return _as_feature_tensor(outputs)

    def get_text_embedding(self, text: str) -> list[float]:
        if self.use_transformers:
            if self.processor is not None:
                encoded = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            else:
                encoded = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                features = self._extract_text_features(encoded)
            return _normalize_vector(features[0].detach().cpu().numpy())

        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.astype(np.float32).tolist()

    def get_image_embedding(self, image: Image.Image) -> list[float]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.use_transformers:
            if self.processor is not None:
                encoded = self.processor(images=image, return_tensors="pt")
            else:
                encoded = self.image_processor(images=image, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                features = self._extract_image_features(encoded)
            return _normalize_vector(features[0].detach().cpu().numpy())

        embedding = self.model.encode(image, normalize_embeddings=True)
        return embedding.astype(np.float32).tolist()