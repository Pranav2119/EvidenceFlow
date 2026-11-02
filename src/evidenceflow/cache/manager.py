"""
manager.py: Orchestrates the 5-layer scoped cache hierarchy.
Provides thread-safe in-memory LRU caching, vector semantic search,
and cross-version invalidation mechanisms.
"""

from collections import OrderedDict
import math
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from evidenceflow.cache.key_generator import (
    generate_claim_key,
    generate_embedding_key,
    generate_retrieval_key,
    generate_scope_key,
    normalize_query,
)
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


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Computes cosine similarity between two float vectors using numpy."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class LRUCache(OrderedDict):
    """Simple thread-safe bounded LRU dictionary."""
    def __init__(self, maxsize: int = 2000):
        super().__init__()
        self.maxsize = maxsize

    def get_and_touch(self, key: str) -> Any:
        if key not in self:
            return None
        self.move_to_end(key)
        return self[key]

    def put(self, key: str, value: Any) -> None:
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.maxsize:
            self.popitem(last=False)  # Evict oldest


class ScopedCacheManager:
    """
    Central 5-Layer Scoped Cache Manager:
      L1: Exact Match (Composite HMAC-SHA256 scope key -> Verified Response)
      L2: Semantic Match (Embedding similarity >= 0.96 within Version Scope)
      L3: Retrieval Match (Query + Version -> Passage IDs & Scores)
      L4: Embedding Cache (Text + Model -> Dense Vector)
      L5: Verified-Claim Cache ((Entity, Attribute, Version) -> Factual Claim)
    """

    def __init__(
        self,
        max_exact_entries: int = 2000,
        max_semantic_entries: int = 1000,
        max_retrieval_entries: int = 2000,
        max_embedding_entries: int = 5000,
        semantic_threshold: float = 0.96,
        default_config_version: str = "v1.0",
        default_guardrail_version: str = "v1.0",
        secret_salt: str = "evidenceflow-salt",
    ):
        self._lock = threading.RLock()
        self.semantic_threshold = semantic_threshold
        self.default_config_version = default_config_version
        self.default_guardrail_version = default_guardrail_version
        self.secret_salt = secret_salt

        # In-memory stores
        self._l1_exact = LRUCache(maxsize=max_exact_entries)
        self._l2_semantic: List[SemanticCacheEntry] = []
        self._l2_max_size = max_semantic_entries
        self._l3_retrieval = LRUCache(maxsize=max_retrieval_entries)
        self._l4_embedding = LRUCache(maxsize=max_embedding_entries)
        self._l5_claim: Dict[str, VerifiedClaimEntry] = {}

        # Telemetry stats
        self.stats = {
            "l1_exact": {"hits": 0, "misses": 0},
            "l2_semantic": {"hits": 0, "misses": 0},
            "l3_retrieval": {"hits": 0, "misses": 0},
            "l4_embedding": {"hits": 0, "misses": 0},
            "l5_claim": {"hits": 0, "misses": 0},
            "evictions": 0,
        }

    # =========================================================================
    # High-Level Orchestrated Lookup
    # =========================================================================

    def lookup(
        self,
        query: str,
        target_versions: Union[Sequence[str], str, None] = "16",
        query_embedding: Optional[Sequence[float]] = None,
        config_version: Optional[str] = None,
        guardrail_version: Optional[str] = None,
        intent: str = "query",
    ) -> CacheLookupResult:
        """
        Orchestrated lookup through L1 (Exact) -> L2 (Semantic).
        Returns a structured CacheLookupResult with timing.
        """
        start_time = time.perf_counter()
        cfg_v = config_version or self.default_config_version
        grd_v = guardrail_version or self.default_guardrail_version

        # 1. Check Layer 1 (Exact Match)
        l1_entry = self.get_exact(
            query=query,
            target_versions=target_versions,
            config_version=cfg_v,
            guardrail_version=grd_v,
            intent=intent,
        )
        if l1_entry is not None:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return CacheLookupResult(
                hit=True,
                layer=CacheLayer.L1_EXACT,
                answer=l1_entry.answer,
                citations=l1_entry.citations,
                similarity_score=1.0,
                metadata={"key": l1_entry.key, "hit_count": l1_entry.hit_count},
                lookup_latency_ms=elapsed,
            )

        # 2. Check Layer 2 (Semantic Match) if embedding provided
        if query_embedding is not None:
            l2_match = self.get_semantic(
                query_embedding=query_embedding,
                target_versions=target_versions,
                config_version=cfg_v,
                guardrail_version=grd_v,
                intent=intent,
            )
            if l2_match is not None:
                l2_entry, score = l2_match
                elapsed = (time.perf_counter() - start_time) * 1000.0
                return CacheLookupResult(
                    hit=True,
                    layer=CacheLayer.L2_SEMANTIC,
                    answer=l2_entry.answer,
                    citations=l2_entry.citations,
                    similarity_score=score,
                    metadata={"id": l2_entry.id, "hit_count": l2_entry.hit_count},
                    lookup_latency_ms=elapsed,
                )

        elapsed = (time.perf_counter() - start_time) * 1000.0
        return CacheLookupResult(hit=False, lookup_latency_ms=elapsed)

    # =========================================================================
    # Layer 1: Exact Match Cache
    # =========================================================================

    def get_exact(
        self,
        query: str,
        target_versions: Union[Sequence[str], str, None] = "16",
        config_version: Optional[str] = None,
        guardrail_version: Optional[str] = None,
        intent: str = "query",
    ) -> Optional[ExactCacheEntry]:
        """Look up an exact match in Layer 1 by composite HMAC scope key."""
        key = generate_scope_key(
            query=query,
            target_versions=target_versions,
            config_version=config_version or self.default_config_version,
            guardrail_version=guardrail_version or self.default_guardrail_version,
            secret_salt=self.secret_salt,
            intent=intent,
        )

        with self._lock:
            entry: Optional[ExactCacheEntry] = self._l1_exact.get_and_touch(key)
            if entry is not None:
                entry.hit_count += 1
                entry.last_accessed = time.time()
                self.stats["l1_exact"]["hits"] += 1
                return entry
            self.stats["l1_exact"]["misses"] += 1
            return None

    def set_exact(
        self,
        query: str,
        answer: str,
        citations: List[CachedCitation],
        target_versions: Union[Sequence[str], str, None] = "16",
        config_version: Optional[str] = None,
        guardrail_version: Optional[str] = None,
        intent: str = "query",
    ) -> ExactCacheEntry:
        """Store a verified answer in Layer 1 with composite scope key."""
        cfg_v = config_version or self.default_config_version
        grd_v = guardrail_version or self.default_guardrail_version

        key = generate_scope_key(
            query=query,
            target_versions=target_versions,
            config_version=cfg_v,
            guardrail_version=grd_v,
            secret_salt=self.secret_salt,
            intent=intent,
        )

        # Normalize versions list
        if target_versions is None:
            v_list = ["any"]
        elif isinstance(target_versions, str):
            v_list = [target_versions.strip()]
        else:
            v_list = [str(v).strip() for v in target_versions if str(v).strip()]

        entry = ExactCacheEntry(
            key=key,
            query=query,
            normalized_query=normalize_query(query),
            target_versions=v_list,
            config_version=cfg_v,
            guardrail_version=grd_v,
            intent=intent,
            answer=answer,
            citations=citations,
        )

        with self._lock:
            self._l1_exact.put(key, entry)

        return entry

    # =========================================================================
    # Layer 2: Semantic Match Cache
    # =========================================================================

    def get_semantic(
        self,
        query_embedding: Sequence[float],
        target_versions: Union[Sequence[str], str, None] = "16",
        config_version: Optional[str] = None,
        guardrail_version: Optional[str] = None,
        intent: str = "query",
        threshold: Optional[float] = None,
    ) -> Optional[Tuple[SemanticCacheEntry, float]]:
        """
        Searches Layer 2 semantic cache for closest vector with matching version scope.
        Requires cosine similarity >= threshold (default 0.96).
        """
        cutoff = threshold if threshold is not None else self.semantic_threshold
        cfg_v = config_version or self.default_config_version
        grd_v = guardrail_version or self.default_guardrail_version

        # Target version set for filtering
        if target_versions is None:
            v_filter = {"any"}
        elif isinstance(target_versions, str):
            v_filter = {target_versions.strip()}
        else:
            v_filter = {str(v).strip() for v in target_versions if str(v).strip()}

        best_entry: Optional[SemanticCacheEntry] = None
        best_score = -1.0

        with self._lock:
            for entry in self._l2_semantic:
                # 1. Metadata filter: Version scope must overlap
                entry_versions = set(entry.target_versions)
                if not (v_filter & entry_versions or "any" in v_filter or "any" in entry_versions):
                    continue
                # 2. Config & guardrail filter: Must match runtime
                if entry.config_version != cfg_v or entry.guardrail_version != grd_v:
                    continue
                # 3. Intent filter
                if entry.intent != intent:
                    continue

                # 4. Vector cosine similarity
                score = _cosine_similarity(query_embedding, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_entry is not None and best_score >= cutoff:
                best_entry.hit_count += 1
                best_entry.last_accessed = time.time()
                self.stats["l2_semantic"]["hits"] += 1
                return (best_entry, best_score)

            self.stats["l2_semantic"]["misses"] += 1
            return None

    def set_semantic(
        self,
        query: str,
        query_embedding: Sequence[float],
        answer: str,
        citations: List[CachedCitation],
        target_versions: Union[Sequence[str], str, None] = "16",
        config_version: Optional[str] = None,
        guardrail_version: Optional[str] = None,
        intent: str = "query",
    ) -> SemanticCacheEntry:
        """Store a verified answer and embedding in Layer 2."""
        cfg_v = config_version or self.default_config_version
        grd_v = guardrail_version or self.default_guardrail_version

        if target_versions is None:
            v_list = ["any"]
        elif isinstance(target_versions, str):
            v_list = [target_versions.strip()]
        else:
            v_list = [str(v).strip() for v in target_versions if str(v).strip()]

        entry_id = f"sem_{time.time_ns()}"
        entry = SemanticCacheEntry(
            id=entry_id,
            query=query,
            normalized_query=normalize_query(query),
            embedding=list(query_embedding),
            target_versions=v_list,
            intent=intent,
            config_version=cfg_v,
            guardrail_version=grd_v,
            answer=answer,
            citations=citations,
        )

        with self._lock:
            self._l2_semantic.append(entry)
            # Evict oldest if beyond capacity
            if len(self._l2_semantic) > self._l2_max_size:
                self._l2_semantic.pop(0)

        return entry

    # =========================================================================
    # Layer 3: Retrieval Cache
    # =========================================================================

    def get_retrieval(
        self,
        query: str,
        target_versions: Union[Sequence[str], str, None] = "16",
        top_k: int = 5,
    ) -> Optional[RetrievalCacheEntry]:
        """Look up cached passage IDs and scores for query + version."""
        key = generate_retrieval_key(
            query=query,
            target_versions=target_versions,
            top_k=top_k,
            secret_salt=self.secret_salt,
        )

        with self._lock:
            entry: Optional[RetrievalCacheEntry] = self._l3_retrieval.get_and_touch(key)
            if entry is not None:
                entry.hit_count += 1
                self.stats["l3_retrieval"]["hits"] += 1
                return entry
            self.stats["l3_retrieval"]["misses"] += 1
            return None

    def set_retrieval(
        self,
        query: str,
        passage_ids: List[str],
        scores: List[float],
        target_versions: Union[Sequence[str], str, None] = "16",
        top_k: int = 5,
    ) -> RetrievalCacheEntry:
        """Cache retrieval results to bypass vector + BM25 search."""
        key = generate_retrieval_key(
            query=query,
            target_versions=target_versions,
            top_k=top_k,
            secret_salt=self.secret_salt,
        )

        if target_versions is None:
            v_list = ["any"]
        elif isinstance(target_versions, str):
            v_list = [target_versions.strip()]
        else:
            v_list = [str(v).strip() for v in target_versions if str(v).strip()]

        entry = RetrievalCacheEntry(
            key=key,
            query=query,
            target_versions=v_list,
            top_k=top_k,
            passage_ids=passage_ids,
            scores=scores,
        )

        with self._lock:
            self._l3_retrieval.put(key, entry)

        return entry

    # =========================================================================
    # Layer 4: Embedding Cache
    # =========================================================================

    def get_embedding(
        self,
        text: str,
        model_name: str = "text-embedding-004",
    ) -> Optional[List[float]]:
        """Look up dense vector for text to eliminate redundant embedding calls."""
        key = generate_embedding_key(text=text, model_name=model_name)

        with self._lock:
            entry: Optional[EmbeddingCacheEntry] = self._l4_embedding.get_and_touch(key)
            if entry is not None:
                entry.hit_count += 1
                self.stats["l4_embedding"]["hits"] += 1
                return entry.embedding
            self.stats["l4_embedding"]["misses"] += 1
            return None

    def set_embedding(
        self,
        text: str,
        embedding: Sequence[float],
        model_name: str = "text-embedding-004",
    ) -> EmbeddingCacheEntry:
        """Cache text embedding."""
        key = generate_embedding_key(text=text, model_name=model_name)
        preview = text[:60] + "..." if len(text) > 60 else text

        entry = EmbeddingCacheEntry(
            key=key,
            text_preview=preview,
            model_name=model_name,
            embedding=list(embedding),
        )

        with self._lock:
            self._l4_embedding.put(key, entry)

        return entry

    # =========================================================================
    # Layer 5: Verified-Claim Cache
    # =========================================================================

    def get_claim(
        self,
        entity: str,
        attribute: str,
        version: str,
    ) -> Optional[VerifiedClaimEntry]:
        """Look up atomic verified claim (e.g. setting default) by (entity, attribute, version)."""
        key = generate_claim_key(entity=entity, attribute=attribute, version=version)

        with self._lock:
            entry = self._l5_claim.get(key)
            if entry is not None:
                entry.hit_count += 1
                self.stats["l5_claim"]["hits"] += 1
                return entry
            self.stats["l5_claim"]["misses"] += 1
            return None

    def set_claim(
        self,
        entity: str,
        attribute: str,
        version: str,
        claim_text: str,
        passage_id: str,
        chunk_hash: str,
    ) -> VerifiedClaimEntry:
        """Store an atomic verified claim for instant factual validation."""
        key = generate_claim_key(entity=entity, attribute=attribute, version=version)

        entry = VerifiedClaimEntry(
            key=key,
            entity=entity.strip().lower(),
            attribute=attribute.strip().lower(),
            version=str(version).strip(),
            claim_text=claim_text,
            passage_id=passage_id,
            chunk_hash=chunk_hash,
        )

        with self._lock:
            self._l5_claim[key] = entry

        return entry

    # =========================================================================
    # Invalidation & Telemetry
    # =========================================================================

    def invalidate_by_version(self, version: str) -> int:
        """
        Evicts all entries across L1, L2, L3, and L5 that reference the given version.
        Guarantees zero stale responses after corpus version re-indexing.
        """
        v_str = str(version).strip()
        evicted = 0

        with self._lock:
            # 1. Invalidate L1
            l1_keys_to_del = [
                k for k, entry in self._l1_exact.items()
                if v_str in entry.target_versions
            ]
            for k in l1_keys_to_del:
                del self._l1_exact[k]
                evicted += 1

            # 2. Invalidate L2
            original_l2_len = len(self._l2_semantic)
            self._l2_semantic = [
                entry for entry in self._l2_semantic
                if v_str not in entry.target_versions
            ]
            evicted += (original_l2_len - len(self._l2_semantic))

            # 3. Invalidate L3
            l3_keys_to_del = [
                k for k, entry in self._l3_retrieval.items()
                if v_str in entry.target_versions
            ]
            for k in l3_keys_to_del:
                del self._l3_retrieval[k]
                evicted += 1

            # 4. Invalidate L5
            l5_keys_to_del = [
                k for k, entry in self._l5_claim.items()
                if entry.version == v_str
            ]
            for k in l5_keys_to_del:
                del self._l5_claim[k]
                evicted += 1

            self.stats["evictions"] += evicted

        return evicted

    def invalidate_by_passage(self, passage_id: str) -> int:
        """
        Evicts all L1, L2, and L5 entries that cite the given passage_id.
        Triggered when a document chunk is updated or deleted.
        """
        evicted = 0

        with self._lock:
            # Invalidate L1 entries citing this passage
            l1_keys = [
                k for k, entry in self._l1_exact.items()
                if any(c.passage_id == passage_id for c in entry.citations)
            ]
            for k in l1_keys:
                del self._l1_exact[k]
                evicted += 1

            # Invalidate L2 entries citing this passage
            orig_len = len(self._l2_semantic)
            self._l2_semantic = [
                entry for entry in self._l2_semantic
                if not any(c.passage_id == passage_id for c in entry.citations)
            ]
            evicted += (orig_len - len(self._l2_semantic))

            # Invalidate L5 entries
            l5_keys = [
                k for k, entry in self._l5_claim.items()
                if entry.passage_id == passage_id
            ]
            for k in l5_keys:
                del self._l5_claim[k]
                evicted += 1

            self.stats["evictions"] += evicted

        return evicted

    def clear_all(self) -> None:
        """Clears all 5 cache layers."""
        with self._lock:
            self._l1_exact.clear()
            self._l2_semantic.clear()
            self._l3_retrieval.clear()
            self._l4_embedding.clear()
            self._l5_claim.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Returns deep telemetry counters and current sizing."""
        with self._lock:
            return {
                "sizes": {
                    "l1_exact": len(self._l1_exact),
                    "l2_semantic": len(self._l2_semantic),
                    "l3_retrieval": len(self._l3_retrieval),
                    "l4_embedding": len(self._l4_embedding),
                    "l5_claim": len(self._l5_claim),
                },
                "telemetry": dict(self.stats),
            }
