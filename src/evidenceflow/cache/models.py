"""
models.py: Pydantic schemas and dataclasses for the 5-layer scoped cache.
"""

from enum import Enum
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CacheLayer(str, Enum):
    """Enumeration of the 5 cache layers in EvidenceFlow AI."""
    L1_EXACT = "L1_EXACT"
    L2_SEMANTIC = "L2_SEMANTIC"
    L3_RETRIEVAL = "L3_RETRIEVAL"
    L4_EMBEDDING = "L4_EMBEDDING"
    L5_CLAIM = "L5_CLAIM"


class CachedCitation(BaseModel):
    """Citation metadata attached to cached answers for deterministic verification."""
    passage_id: str
    chunk_hash: str
    source_url: str = ""
    title: str = ""


class ExactCacheEntry(BaseModel):
    """Layer 1: Exact match cache entry."""
    key: str
    query: str
    normalized_query: str
    target_versions: List[str]
    config_version: str
    guardrail_version: str
    intent: str
    answer: str
    citations: List[CachedCitation]
    created_at: float = Field(default_factory=time.time)
    hit_count: int = 0
    last_accessed: float = Field(default_factory=time.time)


class SemanticCacheEntry(BaseModel):
    """Layer 2: Semantic vector cache entry."""
    id: str
    query: str
    normalized_query: str
    embedding: List[float]
    target_versions: List[str]
    intent: str
    config_version: str
    guardrail_version: str
    answer: str
    citations: List[CachedCitation]
    created_at: float = Field(default_factory=time.time)
    hit_count: int = 0
    last_accessed: float = Field(default_factory=time.time)


class RetrievalCacheEntry(BaseModel):
    """Layer 3: Retrieval results cache entry."""
    key: str
    query: str
    target_versions: List[str]
    top_k: int
    passage_ids: List[str]
    scores: List[float]
    created_at: float = Field(default_factory=time.time)
    hit_count: int = 0


class EmbeddingCacheEntry(BaseModel):
    """Layer 4: Dense embedding vector cache entry."""
    key: str
    text_preview: str
    model_name: str
    embedding: List[float]
    created_at: float = Field(default_factory=time.time)
    hit_count: int = 0


class VerifiedClaimEntry(BaseModel):
    """Layer 5: Atomic verified claim cache entry."""
    key: str
    entity: str
    attribute: str
    version: str
    claim_text: str
    passage_id: str
    chunk_hash: str
    created_at: float = Field(default_factory=time.time)
    hit_count: int = 0


class CacheLookupResult(BaseModel):
    """Result of querying the multi-layer cache hierarchy."""
    hit: bool
    layer: Optional[CacheLayer] = None
    answer: Optional[str] = None
    citations: List[CachedCitation] = Field(default_factory=list)
    similarity_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    lookup_latency_ms: float = 0.0
