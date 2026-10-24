"""
embeddings.py: Dual-provider embedding generator supporting local FastEmbed and Gemini API.
"""

from typing import List
from evidenceflow.config import get_settings


# --- FastEmbed lazy singleton (unchanged) ---
_fastembed_model = None


def _get_fastembed_model():
    global _fastembed_model
    if _fastembed_model is None:
        from fastembed import TextEmbedding
        _fastembed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _fastembed_model


# --- Gemini client lazy singleton (D2-1: avoids re-creating HTTP client per batch) ---
_gemini_client = None

# Gemini embedding API safe sub-batch size (D2-3: API has per-request content limits)
_GEMINI_EMBED_BATCH_SIZE = 20


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        settings = get_settings()
        _gemini_client = genai.Client(api_key=settings.require_gemini_key())
    return _gemini_client


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
        # D2-1: reuse cached client; D2-2: model from config; D2-3: sub-batch to stay under API limits
        client = _get_gemini_client()
        model_name = settings.GEMINI_EMBEDDING_MODEL
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), _GEMINI_EMBED_BATCH_SIZE):
            sub_batch = texts[i : i + _GEMINI_EMBED_BATCH_SIZE]
            response = client.models.embed_content(
                model=model_name,
                contents=sub_batch,
            )
            all_embeddings.extend(item.values for item in response.embeddings)
        return all_embeddings

    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")


def get_embedding(text: str) -> List[float]:
    """Generates embedding for a single text."""
    return get_embeddings([text])[0]
