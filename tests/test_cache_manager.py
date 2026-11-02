"""
test_cache_manager.py: Unit tests for ScopedCacheManager.
Tests L1 Exact Match, L2 Semantic Cache, L3 Retrieval, L4 Embedding, L5 Claim,
and cross-version / passage invalidations.
"""

import numpy as np
import pytest
from evidenceflow.cache.manager import ScopedCacheManager
from evidenceflow.cache.models import CachedCitation, CacheLayer


@pytest.fixture
def cache_mgr():
    """Provides a fresh ScopedCacheManager for each test."""
    return ScopedCacheManager(
        max_exact_entries=10,
        max_semantic_entries=10,
        max_retrieval_entries=10,
        max_embedding_entries=10,
        semantic_threshold=0.95,
    )


def test_l1_exact_match_hit_and_miss(cache_mgr):
    """L1 exact match hits on identical query+version and misses on different version."""
    citations = [
        CachedCitation(passage_id="pg16-wal-archive-mode", chunk_hash="hash123", source_url="https://pg/16")
    ]
    cache_mgr.set_exact(
        query="What is archive_mode default in PostgreSQL 16?",
        answer="In PostgreSQL 16, archive_mode defaults to off.",
        citations=citations,
        target_versions="16",
    )

    # Exact hit
    hit = cache_mgr.get_exact(
        query="what is archive_mode default in postgresql 16?",
        target_versions="16",
    )
    assert hit is not None
    assert hit.answer == "In PostgreSQL 16, archive_mode defaults to off."
    assert hit.hit_count == 1
    assert len(hit.citations) == 1
    assert hit.citations[0].passage_id == "pg16-wal-archive-mode"

    # Miss on different version
    miss = cache_mgr.get_exact(
        query="What is archive_mode default in PostgreSQL 16?",
        target_versions="14",
    )
    assert miss is None


def test_l1_lru_eviction(cache_mgr):
    """L1 respects max_exact_entries capacity and evicts oldest (LRU) items.

    No get() calls are interleaved below, so insertion order == LRU order.
    Queries 0 and 1 are evicted first when capacity (10) is exceeded by 12 inserts.
    """
    for i in range(12):
        cache_mgr.set_exact(
            query=f"Query {i}",
            answer=f"Answer {i}",
            citations=[],
            target_versions="16",
        )

    # Queries 0 and 1 were inserted first and never accessed → evicted (LRU)
    assert cache_mgr.get_exact("Query 0", "16") is None
    assert cache_mgr.get_exact("Query 1", "16") is None
    # Latest query must be present
    assert cache_mgr.get_exact("Query 11", "16") is not None


def test_l2_semantic_match_and_version_filtering(cache_mgr):
    """L2 semantic cache matches high cosine similarity within version scope, rejects cross-version."""
    base_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    close_vec = np.array([0.99, 0.05, 0.0, 0.0], dtype=np.float32)
    diff_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    cache_mgr.set_semantic(
        query="How to configure autovacuum naptime in PG 16?",
        query_embedding=base_vec.tolist(),
        answer="autovacuum_naptime specifies the delay between runs.",
        citations=[CachedCitation(passage_id="p1", chunk_hash="h1")],
        target_versions="16",
    )

    # Similar query within same version -> HIT
    res = cache_mgr.get_semantic(
        query_embedding=close_vec.tolist(),
        target_versions="16",
    )
    assert res is not None
    entry, score = res
    assert score >= 0.95
    assert entry.answer == "autovacuum_naptime specifies the delay between runs."

    # Dissimilar query -> MISS
    res_diff = cache_mgr.get_semantic(
        query_embedding=diff_vec.tolist(),
        target_versions="16",
    )
    assert res_diff is None

    # Similar query but DIFFERENT version -> MISS (strict version isolation)
    res_v14 = cache_mgr.get_semantic(
        query_embedding=close_vec.tolist(),
        target_versions="14",
    )
    assert res_v14 is None


def test_l3_retrieval_cache(cache_mgr):
    """L3 caches passage IDs and scores."""
    cache_mgr.set_retrieval(
        query="shared_buffers config",
        passage_ids=["pg16-res-sb", "pg16-res-mb"],
        scores=[0.92, 0.84],
        target_versions="16",
        top_k=5,
    )

    hit = cache_mgr.get_retrieval(
        query="  shared_buffers config  ",
        target_versions="16",
        top_k=5,
    )
    assert hit is not None
    assert hit.passage_ids == ["pg16-res-sb", "pg16-res-mb"]
    assert hit.scores == [0.92, 0.84]

    # Miss on different top_k
    assert cache_mgr.get_retrieval("shared_buffers config", "16", top_k=10) is None


def test_l4_embedding_cache(cache_mgr):
    """L4 stores and retrieves dense embedding vectors."""
    vec = [0.123, 0.456, 0.789]
    cache_mgr.set_embedding(
        text="Sample document chunk text",
        embedding=vec,
        model_name="text-embedding-004",
    )

    hit = cache_mgr.get_embedding(
        text="Sample document chunk text",
        model_name="text-embedding-004",
    )
    assert hit == vec

    # Miss on different model
    assert cache_mgr.get_embedding("Sample document chunk text", "other-model") is None


def test_l5_verified_claim_cache(cache_mgr):
    """L5 stores atomic parameter default facts."""
    cache_mgr.set_claim(
        entity="archive_mode",
        attribute="default",
        version="16",
        claim_text="off",
        passage_id="pg16-wal-archive-mode",
        chunk_hash="hash999",
    )

    hit = cache_mgr.get_claim(
        entity="ARCHIVE_MODE",
        attribute="DEFAULT",
        version="16",
    )
    assert hit is not None
    assert hit.claim_text == "off"
    assert hit.passage_id == "pg16-wal-archive-mode"


def test_invalidate_by_version(cache_mgr):
    """Invalidating a version removes all entries targeting that version across L1, L2, L3, and L5."""
    # Seed v14 and v16 entries
    cache_mgr.set_exact("Q14", "A14", [], "14")
    cache_mgr.set_exact("Q16", "A16", [], "16")

    cache_mgr.set_semantic("Q14 sem", [1.0, 0.0], "A14 sem", [], "14")
    cache_mgr.set_semantic("Q16 sem", [0.0, 1.0], "A16 sem", [], "16")

    cache_mgr.set_retrieval("Q14 ret", ["p14"], [0.9], "14")
    cache_mgr.set_retrieval("Q16 ret", ["p16"], [0.9], "16")

    cache_mgr.set_claim("param14", "default", "14", "val14", "p14", "h14")
    cache_mgr.set_claim("param16", "default", "16", "val16", "p16", "h16")

    # Invalidate version 14
    evicted = cache_mgr.invalidate_by_version("14")
    assert evicted == 4, f"Expected exactly 4 v14 entries evicted (1xL1+1xL2+1xL3+1xL5), got {evicted}"

    # Verify v14 is gone, v16 remains
    assert cache_mgr.get_exact("Q14", "14") is None
    assert cache_mgr.get_exact("Q16", "16") is not None

    assert cache_mgr.get_semantic([1.0, 0.0], "14") is None
    assert cache_mgr.get_semantic([0.0, 1.0], "16") is not None

    assert cache_mgr.get_retrieval("Q14 ret", "14") is None
    assert cache_mgr.get_retrieval("Q16 ret", "16") is not None

    assert cache_mgr.get_claim("param14", "default", "14") is None
    assert cache_mgr.get_claim("param16", "default", "16") is not None


def test_invalidate_by_passage(cache_mgr):
    """Invalidating a passage ID evicts all L1, L2, and L5 entries citing it."""
    c_p1 = [CachedCitation(passage_id="passage_alpha", chunk_hash="h_alpha")]
    c_p2 = [CachedCitation(passage_id="passage_beta", chunk_hash="h_beta")]

    cache_mgr.set_exact("Q_alpha", "Ans alpha", c_p1, "16")
    cache_mgr.set_exact("Q_beta", "Ans beta", c_p2, "16")

    cache_mgr.set_claim("param_a", "default", "16", "val", "passage_alpha", "h_alpha")
    cache_mgr.set_claim("param_b", "default", "16", "val", "passage_beta", "h_beta")

    # Invalidate passage_alpha
    evicted = cache_mgr.invalidate_by_passage("passage_alpha")
    assert evicted >= 2

    # Alpha gone, Beta remains
    assert cache_mgr.get_exact("Q_alpha", "16") is None
    assert cache_mgr.get_exact("Q_beta", "16") is not None

    assert cache_mgr.get_claim("param_a", "default", "16") is None
    assert cache_mgr.get_claim("param_b", "default", "16") is not None


def test_high_level_orchestrated_lookup(cache_mgr):
    """lookup() returns L1 hit first, falls back to L2, then misses."""
    citations = [CachedCitation(passage_id="p1", chunk_hash="h1")]
    cache_mgr.set_exact("Exact query", "Exact answer", citations, "16")

    # 1. Exact Hit
    res_l1 = cache_mgr.lookup("exact query", "16")
    assert res_l1.hit is True
    assert res_l1.layer == CacheLayer.L1_EXACT
    assert res_l1.answer == "Exact answer"
    assert res_l1.lookup_latency_ms >= 0.0

    # 2. Semantic Hit (when exact misses)
    query_vec = [1.0, 0.0, 0.0]
    cache_mgr.set_semantic(
        query="Semantic base query",
        query_embedding=query_vec,
        answer="Semantic answer",
        citations=citations,
        target_versions="16",
    )

    close_vec = [0.999, 0.01, 0.0]
    res_l2 = cache_mgr.lookup(
        query="Paraphrased different text",
        target_versions="16",
        query_embedding=close_vec,
    )
    assert res_l2.hit is True
    assert res_l2.layer == CacheLayer.L2_SEMANTIC
    assert res_l2.answer == "Semantic answer"

    # 3. Cache Miss
    res_miss = cache_mgr.lookup("Unknown query", "16")
    assert res_miss.hit is False
    assert res_miss.answer is None


def test_invalidate_by_passage_also_evicts_l3(cache_mgr):
    """invalidate_by_passage evicts L3 retrieval entries whose passage_ids list contains the passage."""
    # Seed an L3 entry that includes the target passage in its result set
    cache_mgr.set_retrieval(
        query="archive config",
        passage_ids=["pg16-wal-archive-mode", "pg16-wal-level"],
        scores=[0.91, 0.85],
        target_versions="16",
        top_k=5,
    )
    # Confirm it exists
    assert cache_mgr.get_retrieval("archive config", "16", top_k=5) is not None

    # Invalidate one of the cited passages
    evicted = cache_mgr.invalidate_by_passage("pg16-wal-archive-mode")
    assert evicted >= 1

    # L3 entry must now be gone (stale — referenced passage changed)
    assert cache_mgr.get_retrieval("archive config", "16", top_k=5) is None


def test_get_stats_shape_and_clear_all(cache_mgr):
    """get_stats() returns expected dict shape; clear_all() empties all 5 layers."""
    cache_mgr.set_exact("Q", "A", [], "16")
    cache_mgr.set_semantic("QS", [1.0, 0.0], "AS", [], "16")
    cache_mgr.set_retrieval("QR", ["p1"], [0.9], "16")
    cache_mgr.set_embedding("text chunk", [0.1, 0.2])
    cache_mgr.set_claim("wal_level", "default", "16", "replica", "p1", "h1")

    stats = cache_mgr.get_stats()
    assert "sizes" in stats and "telemetry" in stats
    assert stats["sizes"]["l1_exact"] == 1
    assert stats["sizes"]["l2_semantic"] == 1
    assert stats["sizes"]["l3_retrieval"] == 1
    assert stats["sizes"]["l4_embedding"] == 1
    assert stats["sizes"]["l5_claim"] == 1

    cache_mgr.clear_all()
    after = cache_mgr.get_stats()
    assert all(v == 0 for v in after["sizes"].values())


def test_hit_count_increments_on_l3_l4_l5(cache_mgr):
    """hit_count increments on repeated reads for L3, L4, and L5 layers."""
    # L3
    cache_mgr.set_retrieval("Q3", ["p1"], [0.9], "16")
    hit3a = cache_mgr.get_retrieval("Q3", "16")
    hit3b = cache_mgr.get_retrieval("Q3", "16")
    assert hit3a is not None and hit3b is not None
    assert hit3b.hit_count == 2

    # L4
    cache_mgr.set_embedding("some text", [0.5, 0.5])
    cache_mgr.get_embedding("some text")
    cache_mgr.get_embedding("some text")
    # L4 returns the raw vector list, so check via internal entry
    key = cache_mgr._l4_embedding.get_and_touch(
        __import__("evidenceflow.cache.key_generator", fromlist=["generate_embedding_key"]).generate_embedding_key("some text")
    )
    assert key.hit_count == 2

    # L5
    cache_mgr.set_claim("shared_buffers", "default", "16", "128MB", "p2", "h2")
    cache_mgr.get_claim("shared_buffers", "default", "16")
    hit5 = cache_mgr.get_claim("shared_buffers", "default", "16")
    assert hit5 is not None
    assert hit5.hit_count == 2
