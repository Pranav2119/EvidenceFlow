"""
storage: Storage interfaces for PostgreSQL, Qdrant, and vector collections.
"""

from qdrant_client import QdrantClient as _QdrantClient
from evidenceflow.config import get_settings


def get_qdrant_client() -> _QdrantClient:
    """
    Shared factory returning a QdrantClient configured from settings.
    D2-5: Centralises the 3-parameter construction so a single change covers all callers.
    """
    settings = get_settings()
    return _QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        check_compatibility=False,
    )

