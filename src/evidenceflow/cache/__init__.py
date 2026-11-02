"""
cache package: 5-layer scoped cache hierarchy for EvidenceFlow AI.
"""

from evidenceflow.cache.key_generator import (
    generate_claim_key,
    generate_embedding_key,
    generate_retrieval_key,
    generate_scope_key,
    normalize_query,
)
from evidenceflow.cache.manager import ScopedCacheManager
from evidenceflow.cache.models import (
    CachedCitation,
    CacheLayer,
    CacheLookupResult,
    EmbeddingCacheEntry,
    ExactCacheEntry,
    RetrievalCacheEntry,
    SemanticCacheEntry,
    VerifiedClaimEntry,
)

__all__ = [
    "generate_scope_key",
    "generate_retrieval_key",
    "generate_embedding_key",
    "generate_claim_key",
    "normalize_query",
    "ScopedCacheManager",
    "CacheLayer",
    "CachedCitation",
    "ExactCacheEntry",
    "SemanticCacheEntry",
    "RetrievalCacheEntry",
    "EmbeddingCacheEntry",
    "VerifiedClaimEntry",
    "CacheLookupResult",
]
