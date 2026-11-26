"""
validator.py: Zero-LLM deterministic validation for cache hits in EvidenceFlow AI.
Verifies that cited passage SHA-256 hashes match live corpus records, validates runtime
configuration & guardrail versions, and triggers automatic stale-chunk eviction.
Executes in sub-millisecond time with ZERO LLM calls.
"""

import logging
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

from evidenceflow.cache.manager import ScopedCacheManager
from evidenceflow.cache.models import (
    CachedCitation,
    CacheLookupResult,
    ExactCacheEntry,
    SemanticCacheEntry,
)

logger = logging.getLogger(__name__)


class CorpusDatabase(Protocol):
    """Protocol for fetching live document chunk hashes from storage."""
    def get_chunk_hash(self, passage_id: str) -> Optional[str]:
        """Fetch current SHA-256 hash of a passage from live corpus storage."""
        ...


class InMemoryCorpusDatabase:
    """
    In-memory mock database for unit testing and offline benchmarking.
    Allows dynamic mutation to simulate documentation patches and deletions.
    """
    def __init__(self, chunk_hashes: Optional[Dict[str, str]] = None):
        self._hashes: Dict[str, str] = dict(chunk_hashes or {})

    def set_chunk_hash(self, passage_id: str, sha256_hash: str) -> None:
        self._hashes[passage_id] = sha256_hash

    def remove_chunk(self, passage_id: str) -> None:
        self._hashes.pop(passage_id, None)

    def get_chunk_hash(self, passage_id: str) -> Optional[str]:
        return self._hashes.get(passage_id)


class PostgresCorpusDatabase:
    """
    Live PostgreSQL corpus database checking `corpus_chunks` table for SHA-256 hashes.
    """
    def __init__(self, connection_pool=None):
        self._pool = connection_pool

    def _get_pool(self):
        if self._pool is None:
            from evidenceflow.storage.ledger import get_connection_pool
            self._pool = get_connection_pool(use_app_role=True)
        return self._pool

    def get_chunk_hash(self, passage_id: str) -> Optional[str]:
        pool = self._get_pool()
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT sha256 FROM corpus_chunks WHERE chunk_id = %s;",
                        (passage_id,),
                    )
                    row = cur.fetchone()
                    if row:
                        return row["sha256"] if isinstance(row, dict) else row[0]
                    return None
        except Exception as e:
            logger.warning("Postgres chunk hash lookup failed for '%s': %s", passage_id, e)
            return None


class QdrantCorpusDatabase:
    """
    Live Qdrant vector store database checking point payloads for SHA-256 hashes.
    Acts as fallback or primary store when chunks are indexed in Qdrant.
    """
    def __init__(self, client=None):
        self._client = client

    def _get_client(self):
        if self._client is None:
            from evidenceflow.storage.qdrant_client import get_evidence_qdrant_client
            self._client = get_evidence_qdrant_client()
        return self._client

    def get_chunk_hash(self, passage_id: str) -> Optional[str]:
        try:
            c = self._get_client()
            res = c.get_passage(passage_id)
            if res and res.get("status") == "success":
                return res.get("sha256")
            return None
        except Exception as e:
            logger.warning("Qdrant chunk hash lookup failed for '%s': %s", passage_id, e)
            return None


class CompositeCorpusDatabase:
    """
    Chains multiple corpus databases (e.g. In-memory -> PostgreSQL -> Qdrant).
    """
    def __init__(self, databases: Sequence[CorpusDatabase]):
        self.databases = list(databases)

    def get_chunk_hash(self, passage_id: str) -> Optional[str]:
        for db in self.databases:
            h = db.get_chunk_hash(passage_id)
            if h is not None:
                return h
        return None


def extract_citation_info(item: Any) -> Tuple[Optional[str], Optional[str]]:
    """Extracts (passage_id, chunk_hash) from various citation schemas or dicts."""
    if isinstance(item, CachedCitation):
        return item.passage_id, item.chunk_hash
    if isinstance(item, dict):
        pid = item.get("passage_id") or item.get("chunk_id")
        h = item.get("chunk_hash") or item.get("sha256")
        return pid, h
    # Generic object attribute access (e.g. Citation)
    pid = getattr(item, "passage_id", None) or getattr(item, "chunk_id", None)
    h = getattr(item, "chunk_hash", None) or getattr(item, "sha256", None)
    return pid, h


def validate_cache_hit(
    cached_citations: Sequence[Union[CachedCitation, Dict[str, Any], Any]],
    db: CorpusDatabase,
    config_version: Optional[str] = None,
    guardrail_version: Optional[str] = None,
    current_config_version: str = "v1.0",
    current_guardrail_version: str = "v1.0",
) -> Tuple[bool, Optional[str], List[str]]:
    """
    Deterministically validates a potential cache hit against the live corpus.
    Executes ZERO LLM calls.

    Returns:
      (is_valid: bool, rejection_reason: Optional[str], stale_passage_ids: List[str])

    Validation checks:
      1. Guardrail version parity (current runtime vs cached).
      2. Config version parity (current runtime vs cached).
      3. Citation non-emptiness (ungrounded answers cannot be cached).
      4. Nonblank string hashes and a live match for EVERY cited passage.
    """
    # 1. Guardrail version check
    if guardrail_version is not None and guardrail_version != current_guardrail_version:
        return False, f"GUARDRAIL_VERSION_MISMATCH: cached={guardrail_version} vs current={current_guardrail_version}", []

    # 2. Config version check
    if config_version is not None and config_version != current_config_version:
        return False, f"CONFIG_VERSION_MISMATCH: cached={config_version} vs current={current_config_version}", []

    # 3. Citation existence check
    if not cached_citations:
        return False, "NO_CITATIONS: cached entry lacks grounded citations", []

    stale_passages = []

    # 4. Deterministic SHA-256 hash validation
    for item in cached_citations:
        pid, cached_hash = extract_citation_info(item)
        if not isinstance(pid, str) or not pid.strip():
            return False, "INVALID_CITATION: passage_id must be a nonblank string", []

        if not isinstance(cached_hash, str) or not cached_hash.strip():
            return False, f"INVALID_CACHED_HASH: passage '{pid}' requires a nonblank stored hash", []

        live_hash = db.get_chunk_hash(pid)

        # Chunk missing in live corpus (deleted/renamed)
        if live_hash is None:
            stale_passages.append(pid)
            return False, f"PASSAGE_NOT_FOUND: passage '{pid}' cannot be confirmed in the live corpus", stale_passages

        if not isinstance(live_hash, str) or not live_hash.strip():
            return False, f"INVALID_LIVE_HASH: passage '{pid}' lacks a usable live hash", [pid]

        # Chunk hash mismatch (documentation patched/updated)
        if live_hash != cached_hash:
            stale_passages.append(pid)
            return (
                False,
                f"HASH_MISMATCH: passage '{pid}' hash changed (cached={cached_hash[:8]}... live={live_hash[:8]}...)",
                stale_passages,
            )

    return True, None, []


def evict_stale_cache_entry(
    cache_manager: ScopedCacheManager,
    stale_passages: Sequence[str],
    cache_key: Optional[str] = None,
) -> int:
    """
    Automatically purges stale entries from all cache layers when a hash mismatch
    or deletion is detected.
    """
    total_evicted = 0
    for pid in stale_passages:
        evicted = cache_manager.invalidate_by_passage(pid)
        total_evicted += evicted
        logger.info("Evicted %d cache entries citing modified/deleted passage '%s'", evicted, pid)

    if cache_key and total_evicted == 0:
        # Explicit key deletion if passage invalidation didn't catch it
        with cache_manager._lock:
            if cache_key in cache_manager._l1_exact:
                del cache_manager._l1_exact[cache_key]
                cache_manager.stats["evictions"] += 1
                total_evicted += 1
                logger.info("Explicitly evicted cache key '%s' from L1", cache_key)

    return total_evicted
