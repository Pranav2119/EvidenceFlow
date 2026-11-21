"""
cache_gatekeeper.py: Zero-LLM deterministic cache gatekeeper node for LangGraph.
Performs sub-millisecond cache lookup across L1 Exact and L2 Semantic layers,
validates chunk SHA-256 hashes against live corpus records deterministically,
and automatically evicts stale entries on document patch or chunk deletion.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from evidenceflow.cache.manager import ScopedCacheManager
from evidenceflow.cache.models import CacheLayer, CacheLookupResult
from evidenceflow.cache.validator import (
    CompositeCorpusDatabase,
    CorpusDatabase,
    InMemoryCorpusDatabase,
    PostgresCorpusDatabase,
    QdrantCorpusDatabase,
    evict_stale_cache_entry,
    validate_cache_hit,
)
from evidenceflow.state import Citation, EvidenceFlowState

logger = logging.getLogger(__name__)

# Module-level singletons (can be overridden for testing)
_global_cache_manager: Optional[ScopedCacheManager] = None
_global_corpus_database: Optional[CorpusDatabase] = None


def get_cache_manager() -> ScopedCacheManager:
    """Returns the shared ScopedCacheManager singleton."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = ScopedCacheManager()
    return _global_cache_manager


def set_cache_manager(cm: Optional[ScopedCacheManager]) -> None:
    """Overrides the ScopedCacheManager singleton (useful for isolated tests)."""
    global _global_cache_manager
    _global_cache_manager = cm


def get_corpus_database() -> CorpusDatabase:
    """Returns the shared CorpusDatabase singleton (Postgres -> Qdrant fallback)."""
    global _global_corpus_database
    if _global_corpus_database is None:
        _global_corpus_database = CompositeCorpusDatabase([
            PostgresCorpusDatabase(),
            QdrantCorpusDatabase(),
        ])
    return _global_corpus_database


def set_corpus_database(db: Optional[CorpusDatabase]) -> None:
    """Overrides the CorpusDatabase singleton (useful for isolated tests)."""
    global _global_corpus_database
    _global_corpus_database = db


def cache_gatekeeper_node(state: EvidenceFlowState) -> Dict[str, Any]:
    """
    Plain-code LangGraph node checking for validated cache hits.
    Guarantees ZERO LLM calls when a verified answer is returned.

    Workflow:
      1. Compute composite scope key and query cache (L1 Exact, then L2 Semantic).
      2. If Miss: Proceed to Planner node.
      3. If Hit: Run deterministic SHA-256 hash validation on every cited chunk.
         - If Valid: Return cached answer + citations directly, bypassing Planner/Retriever/LLM.
         - If Stale: Evict stale entries from cache immediately and route to Planner node.
    """
    start_time = time.perf_counter()
    cm = get_cache_manager()
    db = get_corpus_database()

    target_ver = state.target_version or "16"
    query = state.query
    existing_audit = list(getattr(state, "audit_trail", []) or [])

    # 1. Query Cache Manager
    lookup_result: CacheLookupResult = cm.lookup(
        query=query,
        target_versions=target_ver,
    )

    if not lookup_result.hit or not lookup_result.answer:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info("CacheGatekeeper: MISS for query='%s' (%.2f ms)", query[:50], elapsed_ms)
        audit_entry = {
            "node": "cache_gatekeeper",
            "action": "cache_miss",
            "status": "success",
            "cache_hit": False,
            "layer": None,
            "latency_ms": elapsed_ms,
            "prompt_tokens": 0,
            "candidate_tokens": 0,
            "total_tokens": 0,
        }
        return {
            "cache_hit": False,
            "cached_layer": None,
            "audit_trail": existing_audit + [audit_entry],
        }

    # 2. Deterministic Hash Validation (Plain code, ZERO LLM calls)
    is_valid, rejection_reason, stale_passages = validate_cache_hit(
        cached_citations=lookup_result.citations,
        db=db,
        current_config_version=cm.default_config_version,
        current_guardrail_version=cm.default_guardrail_version,
    )

    if not is_valid:
        # Stale cache detected: document patch or deleted chunk!
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.warning(
            "CacheGatekeeper: REJECTED stale hit (%s) in %.2f ms. Evicting...",
            rejection_reason,
            elapsed_ms,
        )
        evict_stale_cache_entry(
            cache_manager=cm,
            stale_passages=stale_passages,
            cache_key=lookup_result.metadata.get("key"),
        )
        audit_entry = {
            "node": "cache_gatekeeper",
            "action": "cache_rejection",
            "status": "evicted",
            "rejection_reason": rejection_reason,
            "stale_passages": stale_passages,
            "cache_hit": False,
            "layer": None,
            "latency_ms": elapsed_ms,
            "prompt_tokens": 0,
            "candidate_tokens": 0,
            "total_tokens": 0,
        }
        return {
            "cache_hit": False,
            "cached_layer": None,
            "audit_trail": existing_audit + [audit_entry],
        }

    # 3. Valid Cache Hit! Return verified answer with zero LLM tokens
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    layer_name = lookup_result.layer.value if lookup_result.layer else CacheLayer.L1_EXACT.value
    logger.info(
        "CacheGatekeeper: HIT [%s] for query='%s' (%.2f ms, 0 LLM tokens)",
        layer_name,
        query[:50],
        elapsed_ms,
    )

    restored_citations = [
        Citation(
            chunk_id=c.passage_id,
            doc_name=c.source_url or f"pg_{target_ver}_doc",
            section=c.title or "Cached Section",
            quote=c.title or "",
            is_verified=True,
            chunk_hash=c.chunk_hash,
        )
        for c in lookup_result.citations
    ]

    audit_entry = {
        "node": "cache_gatekeeper",
        "action": "cache_hit",
        "status": "success",
        "cache_hit": True,
        "layer": layer_name,
        "latency_ms": elapsed_ms,
        "prompt_tokens": 0,
        "candidate_tokens": 0,
        "total_tokens": 0,
        "metadata": lookup_result.metadata,
    }

    return {
        "draft_answer": lookup_result.answer,
        "citations": restored_citations,
        "l1_passed": True,
        "is_abstained": False,
        "abstention_code": None,
        "abstention_reason": None,
        "cache_hit": True,
        "cached_layer": layer_name,
        "audit_trail": existing_audit + [audit_entry],
    }
