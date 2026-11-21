"""
test_cache_validator.py: Deterministic verification tests for Day 10 Cache Validator & Gatekeeper.
Guarantees ZERO LLM calls on cache hit, instant invalidation on chunk hash mismatch,
and automatic eviction of stale entries across cache layers.
"""

import time
from unittest.mock import MagicMock, patch
import pytest

from evidenceflow.cache.manager import ScopedCacheManager
from evidenceflow.cache.models import CachedCitation, CacheLayer
from evidenceflow.cache.validator import (
    InMemoryCorpusDatabase,
    evict_stale_cache_entry,
    validate_cache_hit,
)
from evidenceflow.graph.nodes.cache_gatekeeper import (
    cache_gatekeeper_node,
    set_cache_manager,
    set_corpus_database,
)
from evidenceflow.graph.workflow import create_evidenceflow_graph, run_pipeline
from evidenceflow.state import Citation, EvidenceFlowState


@pytest.fixture(autouse=True)
def reset_singletons():
    """Ensure a clean cache manager and mock DB for every test."""
    cm = ScopedCacheManager()
    db = InMemoryCorpusDatabase()
    set_cache_manager(cm)
    set_corpus_database(db)
    yield cm, db
    set_cache_manager(None)
    set_corpus_database(None)


def test_exact_hit_validation_success():
    """Valid citations with matching hashes pass validation deterministically."""
    db = InMemoryCorpusDatabase({
        "pg16_max_conn_01": "hash_abc_12345",
        "pg16_shared_buf_02": "hash_def_67890",
    })

    citations = [
        CachedCitation(passage_id="pg16_max_conn_01", chunk_hash="hash_abc_12345"),
        CachedCitation(passage_id="pg16_shared_buf_02", chunk_hash="hash_def_67890"),
    ]

    is_valid, reason, stale = validate_cache_hit(
        cached_citations=citations,
        db=db,
        config_version="v1.0",
        guardrail_version="v1.0",
    )

    assert is_valid is True
    assert reason is None
    assert stale == []


def test_hash_mismatch_staleness_rejection():
    """Simulates documentation patch: live hash != cached hash -> rejection."""
    db = InMemoryCorpusDatabase({
        "pg16_max_conn_01": "hash_NEW_PATCHED_999",  # Modified hash
    })

    citations = [
        CachedCitation(passage_id="pg16_max_conn_01", chunk_hash="hash_OLD_ORIGINAL_111"),
    ]

    is_valid, reason, stale = validate_cache_hit(
        cached_citations=citations,
        db=db,
    )

    assert is_valid is False
    assert "HASH_MISMATCH" in reason
    assert "pg16_max_conn_01" in stale


def test_deleted_chunk_rejection():
    """Chunk removed from live documentation corpus -> rejection."""
    db = InMemoryCorpusDatabase({})  # Empty DB

    citations = [
        CachedCitation(passage_id="pg16_deleted_chunk", chunk_hash="some_hash"),
    ]

    is_valid, reason, stale = validate_cache_hit(
        cached_citations=citations,
        db=db,
    )

    assert is_valid is False
    assert "PASSAGE_NOT_FOUND" in reason
    assert "pg16_deleted_chunk" in stale


def test_no_citations_rejection():
    """Ungrounded answers with zero citations are refused from being served as cache hits."""
    db = InMemoryCorpusDatabase()
    is_valid, reason, stale = validate_cache_hit(
        cached_citations=[],
        db=db,
    )
    assert is_valid is False
    assert "NO_CITATIONS" in reason


def test_config_and_guardrail_version_mismatch():
    """Runtime config/guardrail version change immediately invalidates cache hits."""
    db = InMemoryCorpusDatabase({"chunk1": "hash1"})
    citations = [CachedCitation(passage_id="chunk1", chunk_hash="hash1")]

    # Guardrail mismatch
    is_valid, reason, _ = validate_cache_hit(
        cached_citations=citations,
        db=db,
        guardrail_version="v1.0",
        current_guardrail_version="v2.0",
    )
    assert is_valid is False
    assert "GUARDRAIL_VERSION_MISMATCH" in reason

    # Config mismatch
    is_valid, reason, _ = validate_cache_hit(
        cached_citations=citations,
        db=db,
        config_version="v1.0",
        current_config_version="v1.1",
    )
    assert is_valid is False
    assert "CONFIG_VERSION_MISMATCH" in reason


def test_automatic_stale_eviction():
    """When stale chunks are detected, evict_stale_cache_entry purges all matching cache entries."""
    cm = ScopedCacheManager()
    entry = cm.set_exact(
        query="What is max_connections in v16?",
        answer="Default max_connections is 100 [^chunk1].",
        citations=[CachedCitation(passage_id="chunk1", chunk_hash="h1")],
        target_versions=["16"],
    )

    assert len(cm._l1_exact) == 1

    # Evict stale chunk1
    evicted = evict_stale_cache_entry(cm, stale_passages=["chunk1"], cache_key=entry.key)
    assert evicted >= 1
    assert len(cm._l1_exact) == 0


def test_cache_gatekeeper_node_miss():
    """On cache miss, Gatekeeper sets cache_hit: False, records 0 tokens, and routes to planner."""
    state = EvidenceFlowState(
        query="What is wal_compression in v16?",
        target_version="16",
    )
    res = cache_gatekeeper_node(state)
    assert res["cache_hit"] is False
    assert res["cached_layer"] is None

    audit = res["audit_trail"][-1]
    assert audit["action"] == "cache_miss"
    assert audit["prompt_tokens"] == 0
    assert audit["total_tokens"] == 0


def test_cache_gatekeeper_node_hit():
    """On valid cache hit, Gatekeeper returns answer in <5ms with 0 tokens and restored citations."""
    cm = ScopedCacheManager()
    db = InMemoryCorpusDatabase({
        "chunk_mc_16": "hash_mc_16_valid",
    })
    set_cache_manager(cm)
    set_corpus_database(db)

    # Pre-populate L1 cache
    cm.set_exact(
        query="What is max_connections in v16?",
        answer="In PostgreSQL 16, default max_connections is 100 [^chunk_mc_16].",
        citations=[CachedCitation(passage_id="chunk_mc_16", chunk_hash="hash_mc_16_valid", title="Resource Consumption")],
        target_versions=["16"],
    )

    state = EvidenceFlowState(
        query="What is max_connections in v16?",
        target_version="16",
    )

    t0 = time.perf_counter()
    res = cache_gatekeeper_node(state)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert res["cache_hit"] is True
    assert res["cached_layer"] == "L1_EXACT"
    assert "default max_connections is 100" in res["draft_answer"]
    assert res["l1_passed"] is True
    assert len(res["citations"]) == 1
    assert res["citations"][0].chunk_id == "chunk_mc_16"
    assert res["citations"][0].is_verified is True
    assert elapsed_ms < 50.0  # Sub-50ms (typically <2ms in Python)

    audit = res["audit_trail"][-1]
    assert audit["action"] == "cache_hit"
    assert audit["prompt_tokens"] == 0
    assert audit["candidate_tokens"] == 0
    assert audit["total_tokens"] == 0


def test_cache_gatekeeper_node_stale_rejection():
    """Simulates a doc patch: Gatekeeper catches hash mismatch, evicts entry, and sets cache_hit: False."""
    cm = ScopedCacheManager()
    db = InMemoryCorpusDatabase({
        "chunk_mc_16": "hash_NEW_PATCHED",  # Hash changed!
    })
    set_cache_manager(cm)
    set_corpus_database(db)

    # Pre-populate L1 cache with old hash
    entry = cm.set_exact(
        query="What is max_connections in v16?",
        answer="Old cached answer [^chunk_mc_16].",
        citations=[CachedCitation(passage_id="chunk_mc_16", chunk_hash="hash_OLD", title="Resource Consumption")],
        target_versions=["16"],
    )

    assert entry.key in cm._l1_exact

    state = EvidenceFlowState(
        query="What is max_connections in v16?",
        target_version="16",
    )

    res = cache_gatekeeper_node(state)
    assert res["cache_hit"] is False
    assert res["cached_layer"] is None

    # Verify stale entry was evicted from cache
    assert entry.key not in cm._l1_exact
    audit = res["audit_trail"][-1]
    assert audit["action"] == "cache_rejection"
    assert audit["status"] == "evicted"


@patch("evidenceflow.graph.workflow.pg_record_turn")
@patch("evidenceflow.graph.audit_wrapper.get_next_step_number", return_value=1)
@patch("evidenceflow.graph.audit_wrapper.record_claim")
@patch("evidenceflow.graph.audit_wrapper.record_step")
@patch("evidenceflow.graph.nodes.retriever.get_evidence_qdrant_client")
def test_end_to_end_workflow_two_turn_cache_acceleration(mock_qdrant, mock_record_step, mock_record_claim, mock_get_next_step, mock_pg_turn):
    """
    Two-run test:
      Run 1: Cache Miss -> full pipeline execution -> verifier passes -> cached.
      Run 2: Cache Hit -> Gatekeeper intercepts -> instant answer, cache_hit: True, 0 tokens.
    """
    cm = ScopedCacheManager()
    db = InMemoryCorpusDatabase({
        "pg16_doc_test": "valid_sha256_mock_hash",
    })
    set_cache_manager(cm)
    set_corpus_database(db)

    # Mock Qdrant retrieval response
    mock_client = MagicMock()
    mock_client.search.return_value = [{
        "chunk_id": "pg16_doc_test",
        "sha256": "valid_sha256_mock_hash",
        "version": "16",
        "doc_name": "runtime.html",
        "section": "Resource Consumption",
        "title": "max_connections",
        "content": "max_connections (integer) Sets the maximum number of concurrent connections. The default is 100.",
        "score": 0.95,
    }]
    mock_qdrant.return_value = mock_client

    # Run 1: Fresh run (Cache Miss -> Planner -> Retriever -> AnswerWriter -> Verifier -> Cache Store)
    state1 = run_pipeline(
        query="What is max_connections in v16?",
        version="16",
        enable_memory=False,
        enable_cache=True,
    )

    assert state1.cache_hit is False
    assert state1.l1_passed is True
    assert len(state1.draft_answer) > 0

    # Run 2: Exact repeat query (Cache Hit -> Intercepted at Gatekeeper with 0 LLM calls)
    t0 = time.perf_counter()
    state2 = run_pipeline(
        query="What is max_connections in v16?",
        version="16",
        enable_memory=False,
        enable_cache=True,
    )
    hit_latency_ms = (time.perf_counter() - t0) * 1000.0

    assert state2.cache_hit is True
    assert state2.cached_layer == "L1_EXACT"
    assert state2.draft_answer == state1.draft_answer
    assert state2.l1_passed is True
    assert hit_latency_ms < 50.0  # Instant execution without LLM calls


@patch("evidenceflow.graph.workflow.pg_record_turn")
@patch("evidenceflow.graph.audit_wrapper.get_next_step_number", return_value=1)
@patch("evidenceflow.graph.audit_wrapper.record_claim")
@patch("evidenceflow.graph.audit_wrapper.record_step")
@patch("evidenceflow.graph.nodes.retriever.get_evidence_qdrant_client")
def test_break_it_on_purpose_tampered_hash_eviction(mock_qdrant, mock_record_step, mock_record_claim, mock_get_next_step, mock_pg_turn):
    """
    Break It on Purpose (Roadmap Day 10 Section 6):
    Simulate a documentation patch: artificially change a single character in the source
    chunk's stored SHA-256 hash. Run query again and verify that CacheGatekeeper identifies
    hash mismatch, rejects cache entry, evicts it, and falls back to fresh retrieval.
    """
    cm = ScopedCacheManager()
    original_hash = "235279b3f1c8d212ae254216b3f468260ceb0ff6ca7dd99698c36a1e8d3ab5c7"
    db = InMemoryCorpusDatabase({
        "pg16_chunk_01": original_hash,
    })
    set_cache_manager(cm)
    set_corpus_database(db)

    # Pre-populate cache
    cm.set_exact(
        query="What is max_connections in v16?",
        answer="In PostgreSQL 16, default max_connections is 100 [^pg16_chunk_01].",
        citations=[CachedCitation(passage_id="pg16_chunk_01", chunk_hash=original_hash, title="Resource Consumption")],
        target_versions=["16"],
    )

    # Verify query hits cache initially
    state_before = run_pipeline(
        query="What is max_connections in v16?",
        version="16",
        enable_memory=False,
        enable_cache=True,
    )
    assert state_before.cache_hit is True

    # Sabotage: Artificially mutate the last character of the stored hash in the corpus database
    tampered_hash = original_hash[:-1] + ("0" if original_hash[-1] != "0" else "1")
    db.set_chunk_hash("pg16_chunk_01", tampered_hash)

    # Mock Qdrant fallback retrieval
    mock_client = MagicMock()
    mock_client.search.return_value = [{
        "chunk_id": "pg16_chunk_01",
        "sha256": tampered_hash,
        "version": "16",
        "doc_name": "runtime.html",
        "section": "Resource Consumption",
        "title": "max_connections",
        "content": "max_connections (integer) Sets the maximum number of concurrent connections. The default is 100.",
        "score": 0.95,
    }]
    mock_qdrant.return_value = mock_client

    # Run query again: Gatekeeper must detect hash mismatch, reject cached entry, and fall back to fresh retrieval
    state_after = run_pipeline(
        query="What is max_connections in v16?",
        version="16",
        enable_memory=False,
        enable_cache=True,
    )

    assert state_after.cache_hit is False  # Cache rejected!
    assert state_after.l1_passed is True   # Fresh generation succeeded!

    # Verify the stale entry was purged from L1 cache
    entry_in_cache = cm.get_exact(query="What is max_connections in v16?", target_versions="16")
    # If freshly cached, its citation has the new tampered_hash, not the original_hash
    if entry_in_cache:
        assert entry_in_cache.citations[0].chunk_hash == tampered_hash
