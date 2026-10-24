"""
embeddings.py: Dual-provider embedding generator supporting local FastEmbed and Gemini API.
"""

from typing import List
from evidenceflow.config import get_settings


_fastembed_model = None


def _get_fastembed_model():
    global _fastembed_model
    if _fastembed_model is None:
        from fastembed import TextEmbedding
        _fastembed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _fastembed_model


def get_vector_dimension() -> int:
    """Returns the vector dimensionality for the active provider."""
    settings = get_settings()
    if settings.EMBEDDING_PROVIDER == "fastembed":
        return 384
    elif settings.EMBEDDING_PROVIDER == "gemini":
        return 768
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates dense vector embeddings for a list of texts using the active provider.
    """
    if not texts:
        return []

    settings = get_settings()

    if settings.EMBEDDING_PROVIDER == "fastembed":
        model = _get_fastembed_model()
        # FastEmbed returns an iterator of numpy ndarrays
        return [vec.tolist() for vec in model.embed(texts)]

    elif settings.EMBEDDING_PROVIDER == "gemini":
        api_key = settings.require_gemini_key()
        from google import genai
        client = genai.Client(api_key=api_key)

        # Batch embed via google-genai
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=texts,
        )
        return [item.values for item in response.embeddings]

    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")


def get_embedding(text: str) -> List[float]:
    """Generates embedding for a single text."""
    return get_embeddings([text])[0]
