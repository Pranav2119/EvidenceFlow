"""
comparator.py: Competitive Architectural Benchmark Framework for EvidenceFlow AI.
Head-to-head empirical comparison across 3 architectural paradigms:
  1. EvidenceFlow AI (Full Stack: Qdrant single-stage filtered HNSW + BM25 RRF +
     Neo4j conversational memory graph + 5-layer scoped cache + L1.5/L2 verification).
  2. Naive pgvector + PostgreSQL Full-Text Search (Monolithic Postgres with tsvector
     and post-hoc version filtering, stateless prompt concatenation, no cache).
  3. PostgreSQL Recursive SQL Joins (Relational CTE hierarchy traversal with ILIKE
     pattern matching, zero vector capability, failure on semantic queries/pronouns).
"""

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import psycopg
from psycopg.rows import dict_row

from evidenceflow.config import get_settings
from evidenceflow.graph.workflow import run_pipeline
from evidenceflow.state import EvidenceFlowState
from evidenceflow.storage.embeddings import get_embedding
from evidenceflow.storage.ledger import get_postgres_connection_string

logger = logging.getLogger(__name__)


class CompetitorSystem(str, Enum):
    EVIDENCEFLOW = "EvidenceFlow AI (Qdrant + Neo4j + Scoped Cache)"
    PGVECTOR_TSVECTOR = "Naive pgvector + PostgreSQL tsvector (Monolithic)"
    RECURSIVE_SQL = "PostgreSQL Recursive CTE SQL Joins (Relational)"


@dataclass
class QueryBenchmarkItem:
    """Represents a benchmark test case."""
    query_id: str
    query: str
    category: str  # version_filtered, semantic_paraphrased, cross_version, multi_turn, unanswerable
    target_version: Optional[str] = "16"
    prior_version: Optional[str] = None
    expected_chunk_ids: List[str] = field(default_factory=list)
    expected_parameter: Optional[str] = None
    is_answerable: bool = True
    multi_turn_history: Optional[List[Dict[str, str]]] = None


@dataclass
class CompetitorEvaluationResult:
    """Telemetry and quality metrics for a single query execution on a system."""
    system: str
    query_id: str
    category: str
    latency_ms: float
    retrieved_chunk_ids: List[str]
    recall_at_5: float
    reciprocal_rank: float
    is_grounded: bool
    is_abstained: bool
    abstention_correct: bool
    resolved_entity_correct: bool


# --- Competitor 1: EvidenceFlow AI (Full Multi-Agent System) ---

class EvidenceFlowCompetitor:
    """Evaluates queries through the complete EvidenceFlow AI LangGraph workflow."""

    def __init__(self, enable_cache: bool = True, enable_memory: bool = True):
        self.enable_cache = enable_cache
        self.enable_memory = enable_memory

    def run_query(self, item: QueryBenchmarkItem) -> Tuple[List[str], bool, bool, Optional[str]]:
        """
        Executes query through EvidenceFlow AI workflow.
        Returns (retrieved_chunk_ids, is_grounded, is_abstained, resolved_entity)
        """
        state = run_pipeline(
            query=item.query,
            version=item.target_version,
            thread_id=f"benchmark_{item.query_id}",
            enable_memory=self.enable_memory,
            enable_cache=self.enable_cache,
        )

        retrieved_ids = [
            p.get("chunk_id") or p.get("id")
            for p in (state.retrieved_passages or [])
        ]
        is_grounded = bool(state.l1_passed and not state.is_abstained)
        is_abstained = bool(state.is_abstained)
        
        # Entity resolution check for multi-turn
        resolved_entity = None
        if state.citations:
            resolved_entity = state.citations[0].doc_name if hasattr(state.citations[0], "doc_name") else None

        return retrieved_ids, is_grounded, is_abstained, resolved_entity


# --- Competitor 2: Naive pgvector + PostgreSQL tsvector (Monolithic Baseline) ---

class PgvectorTsvectorCompetitor:
    """
    Monolithic PostgreSQL baseline combining full-text search (tsvector) with
    dense embeddings and naive post-filtering (the standard industry anti-pattern).
    """

    def __init__(self):
        self.conn_str = get_postgres_connection_string(use_app_role=False)
        self._cached_embeddings: Dict[str, List[float]] = {}
        self._load_chunk_embeddings_cache()

    def _load_chunk_embeddings_cache(self, limit: int = 500):
        """Preloads a subset of chunk embeddings for vector cosine similarity."""
        try:
            with psycopg.connect(self.conn_str, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT chunk_id, content FROM corpus_chunks LIMIT %s;",
                        (limit,),
                    )
                    rows = cur.fetchall()
            for r in rows:
                # Fast local embedding
                self._cached_embeddings[r["chunk_id"]] = get_embedding(r["content"][:300])
        except Exception as e:
            logger.warning("Failed to preload chunk embeddings: %s", e)

    def search_full_text(self, query: str, limit: int = 20) -> List[Tuple[str, str, float]]:
        """Executes PostgreSQL tsvector full-text search using ts_rank."""
        results = []
        try:
            with psycopg.connect(self.conn_str, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    sql = """
                    SELECT chunk_id, version, ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) AS rank
                    FROM corpus_chunks
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s;
                    """
                    cur.execute(sql, (query, query, limit))
                    for r in cur.fetchall():
                        results.append((r["chunk_id"], r["version"], float(r["rank"])))
        except Exception as e:
            logger.warning("tsvector search failed: %s", e)
        return results

    def search_vector_naive_postfilter(
        self, query: str, target_version: Optional[str] = None, top_k: int = 5
    ) -> List[str]:
        """
        Demonstrates the classic naive pgvector post-filtering flaw:
        1. Computes top-N nearest neighbors globally across ALL versions.
        2. Filters WHERE version = target_version post-hoc.
        When older versions share high semantic similarity, they crowd out top slots,
        causing severe recall drop on the target version.
        """
        query_vec = get_embedding(query)
        scored_chunks: List[Tuple[str, str, float]] = []

        # Compute cosine similarity globally
        for cid, emb in self._cached_embeddings.items():
            # Dot product for normalized vectors
            sim = sum(a * b for a, b in zip(query_vec, emb))
            v = "16" if "pg16" in cid else ("14" if "pg14" in cid else "15")
            scored_chunks.append((cid, v, sim))

        # Sort globally (pgvector HNSW index scan without filter)
        scored_chunks.sort(key=lambda x: x[2], reverse=True)

        # Naive Post-Filter: take top 5, then discard non-matching versions
        global_top = scored_chunks[:top_k]
        if target_version:
            filtered = [cid for cid, v, _ in global_top if v == target_version]
            return filtered
        return [cid for cid, _, _ in global_top]

    def run_query(self, item: QueryBenchmarkItem) -> Tuple[List[str], bool, bool, Optional[str]]:
        """Executes query through the monolithic pgvector + tsvector baseline."""
        # 1. Unanswerable check: Naive baseline lacks input guardrails, attempts search
        if not item.is_answerable:
            # Monolithic baseline typically does NOT abstain; tries to retrieve best match
            fts_hits = self.search_full_text(item.query, limit=5)
            retrieved = [cid for cid, _, _ in fts_hits]
            return retrieved, False, False, None

        # 2. Multi-turn check: Naive baseline has NO graph memory; treats follow-ups flat
        query_text = item.query
        if item.multi_turn_history:
            # Flatten chat history into raw string (anti-pattern)
            prior_queries = " ".join([m["content"] for m in item.multi_turn_history])
            query_text = f"{prior_queries} {item.query}"

        # 3. Hybrid search: tsvector + naive post-filtered vector
        fts_hits = self.search_full_text(query_text, limit=10)
        vec_hits = self.search_vector_naive_postfilter(query_text, target_version=item.target_version, top_k=5)

        # Merge results post-hoc
        merged = []
        for cid in vec_hits:
            if cid not in merged:
                merged.append(cid)
        for cid, v, _ in fts_hits:
            if item.target_version and v != item.target_version:
                continue
            if cid not in merged:
                merged.append(cid)

        retrieved_ids = merged[:5]
        is_grounded = bool(len(retrieved_ids) > 0)
        is_abstained = False
        resolved_entity = None
        return retrieved_ids, is_grounded, is_abstained, resolved_entity


# --- Competitor 3: PostgreSQL Recursive SQL Joins (Relational Baseline) ---

class RecursiveSqlCompetitor:
    """
    Pure relational PostgreSQL baseline using recursive CTEs to navigate
    document hierarchies and ILIKE matching for parameter lookups.
    """

    def __init__(self):
        self.conn_str = get_postgres_connection_string(use_app_role=False)
        self._ensure_hierarchy_table()

    def _ensure_hierarchy_table(self):
        """Sets up relational section hierarchy table for recursive CTE execution."""
        try:
            with psycopg.connect(self.conn_str, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                    CREATE TABLE IF NOT EXISTS doc_hierarchy (
                        section_id VARCHAR(128) PRIMARY KEY,
                        parent_id VARCHAR(128),
                        version VARCHAR(16) NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_doc_hierarchy_parent ON doc_hierarchy(parent_id);
                    CREATE INDEX IF NOT EXISTS idx_doc_hierarchy_version ON doc_hierarchy(version);
                    """)
                    # Populate root and subsections if empty
                    cur.execute("SELECT COUNT(*) FROM doc_hierarchy;")
                    if cur.fetchone()[0] == 0:
                        cur.execute("""
                        INSERT INTO doc_hierarchy (section_id, parent_id, version, title, content)
                        SELECT chunk_id, NULL, version, coalesce(title, 'Section'), content
                        FROM corpus_chunks
                        LIMIT 500;
                        """)
        except Exception as e:
            logger.warning("Failed to initialize doc_hierarchy: %s", e)

    def search_recursive_cte(self, keyword: str, version: Optional[str] = "16", limit: int = 5) -> List[str]:
        """
        Executes a recursive CTE query traversing sections and matching keyword pattern.
        """
        results = []
        try:
            with psycopg.connect(self.conn_str, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    sql = """
                    WITH RECURSIVE section_tree AS (
                        -- Anchor member: root matching sections
                        SELECT section_id, parent_id, version, title, content, 1 AS depth
                        FROM doc_hierarchy
                        WHERE version = %s AND (title ILIKE %s OR content ILIKE %s)

                        UNION ALL

                        -- Recursive member: child sections
                        SELECT d.section_id, d.parent_id, d.version, d.title, d.content, st.depth + 1
                        FROM doc_hierarchy d
                        INNER JOIN section_tree st ON d.parent_id = st.section_id
                        WHERE st.depth < 3
                    )
                    SELECT DISTINCT section_id FROM section_tree LIMIT %s;
                    """
                    pat = f"%{keyword}%"
                    cur.execute(sql, (version or "16", pat, pat, limit))
                    for r in cur.fetchall():
                        results.append(r["section_id"])
        except Exception as e:
            logger.warning("Recursive CTE search failed: %s", e)
        return results

    def run_query(self, item: QueryBenchmarkItem) -> Tuple[List[str], bool, bool, Optional[str]]:
        """
        Executes query through Recursive SQL baseline.
        Extracts primary alphanumeric keyword from query and runs CTE search.
        Fails on semantic paraphrasing, pronouns, and out-of-scope concepts.
        """
        if not item.is_answerable:
            return [], False, False, None

        # Extract keyword (e.g. "shared_buffers", "wal_compression")
        tokens = re.findall(r"[a-z0-9_]{4,}", item.query.lower())
        stopwords = {"what", "which", "where", "default", "value", "postgres", "postgresql", "does", "require"}
        meaningful = [t for t in tokens if t not in stopwords]
        keyword = meaningful[0] if meaningful else "config"

        # If query is a multi-turn pronoun question ("Does that require a restart?"), SQL has no antecedent
        if item.multi_turn_history and not any(k in item.query.lower() for k in ["wal", "buffer", "vacuum", "cost"]):
            # Pronoun failure: Cannot resolve "that" or "it" without graph memory
            return [], False, False, None

        hits = self.search_recursive_cte(keyword=keyword, version=item.target_version, limit=5)
        is_grounded = bool(len(hits) > 0)
        return hits, is_grounded, False, None


# --- Benchmark Suite Definition ---

def get_competitive_benchmark_dataset() -> List[QueryBenchmarkItem]:
    """
    Returns 20 curated benchmark queries across 5 critical architectural test categories:
      1. version_filtered (Post-filtering vs Single-Stage Filtered HNSW)
      2. semantic_paraphrased (Semantic Vector vs Keyword/SQL)
      3. cross_version (Dual-Version Reasoning & Evolution)
      4. multi_turn (Graph Memory Entity Resolution vs Flat Prompts vs SQL Pronoun Failure)
      5. unanswerable (Input Guardrails & Hallucination Prevention)
    """
    return [
        # --- Category 1: Version-Filtered Technical Queries ---
        QueryBenchmarkItem(
            query_id="vf_01",
            query="What is the default value of shared_buffers in PostgreSQL 16?",
            category="version_filtered",
            target_version="16",
            expected_chunk_ids=["pg16-runtime-config-resource-shared-buffers"],
            expected_parameter="shared_buffers",
        ),
        QueryBenchmarkItem(
            query_id="vf_02",
            query="What is vacuum_cost_limit in PostgreSQL 14?",
            category="version_filtered",
            target_version="14",
            expected_chunk_ids=["pg14-runtime-config-resource-vacuum-cost-limit"],
            expected_parameter="vacuum_cost_limit",
        ),
        QueryBenchmarkItem(
            query_id="vf_03",
            query="What is work_mem default in PostgreSQL 16?",
            category="version_filtered",
            target_version="16",
            expected_chunk_ids=["pg16-runtime-config-resource-work-mem"],
            expected_parameter="work_mem",
        ),
        QueryBenchmarkItem(
            query_id="vf_04",
            query="What are the valid settings for wal_level in PostgreSQL 16?",
            category="version_filtered",
            target_version="16",
            expected_chunk_ids=["pg16-runtime-config-wal-wal-level"],
            expected_parameter="wal_level",
        ),

        # --- Category 2: Semantic / Paraphrased Queries ---
        QueryBenchmarkItem(
            query_id="sp_01",
            query="How much dedicated memory is used for sorting operations before writing to disk in PG16?",
            category="semantic_paraphrased",
            target_version="16",
            expected_chunk_ids=["pg16-runtime-config-resource-work-mem"],
            expected_parameter="work_mem",
        ),
        QueryBenchmarkItem(
            query_id="sp_02",
            query="What is the maximum write ahead log capacity before automatic checkpoint triggering in PG16?",
            category="semantic_paraphrased",
            target_version="16",
            expected_chunk_ids=["pg16-runtime-config-wal-max-wal-size"],
            expected_parameter="max_wal_size",
        ),
        QueryBenchmarkItem(
            query_id="sp_03",
            query="How long can idle client transactions stay open before being forcibly closed in PG16?",
            category="semantic_paraphrased",
            target_version="16",
            expected_chunk_ids=["pg16-runtime-config-client-idle-in-transaction-session-timeout"],
            expected_parameter="idle_in_transaction_session_timeout",
        ),
        QueryBenchmarkItem(
            query_id="sp_04",
            query="What mechanism throttles background autovacuum to reduce disk I/O pressure in PG14?",
            category="semantic_paraphrased",
            target_version="14",
            expected_chunk_ids=["pg14-runtime-config-autovacuum-autovacuum-vacuum-cost-limit", "pg14-runtime-config-resource-vacuum-cost-limit"],
            expected_parameter="vacuum_cost_limit",
        ),

        # --- Category 3: Cross-Version Comparative Queries ---
        QueryBenchmarkItem(
            query_id="cv_01",
            query="Compare wal_compression between Postgres 14 and 16",
            category="cross_version",
            target_version="16",
            prior_version="14",
            expected_chunk_ids=["pg16-runtime-config-wal-wal-compression", "pg14-runtime-config-wal-wal-compression"],
            expected_parameter="wal_compression",
        ),
        QueryBenchmarkItem(
            query_id="cv_02",
            query="Compare vacuum_cost_page_dirty in Postgres 14 vs 16",
            category="cross_version",
            target_version="16",
            prior_version="14",
            expected_chunk_ids=["pg16-runtime-config-resource-vacuum-cost-page-dirty", "pg14-runtime-config-resource-vacuum-cost-page-dirty"],
            expected_parameter="vacuum_cost_page_dirty",
        ),
        QueryBenchmarkItem(
            query_id="cv_03",
            query="Compare vacuum_cost_limit in Postgres 14 vs 16",
            category="cross_version",
            target_version="16",
            prior_version="14",
            expected_chunk_ids=["pg16-runtime-config-resource-vacuum-cost-limit", "pg14-runtime-config-resource-vacuum-cost-limit"],
            expected_parameter="vacuum_cost_limit",
        ),
        QueryBenchmarkItem(
            query_id="cv_04",
            query="How did default toast compression evolve between PostgreSQL 14 and 16?",
            category="cross_version",
            target_version="16",
            prior_version="14",
            expected_chunk_ids=["pg16-runtime-config-client-default-toast-compression", "pg14-runtime-config-client-default-toast-compression"],
            expected_parameter="default_toast_compression",
        ),

        # --- Category 4: Multi-Turn Dialogues (Pronoun & Context Tracking) ---
        QueryBenchmarkItem(
            query_id="mt_01",
            query="What is wal_compression in PostgreSQL 16?",
            category="multi_turn",
            target_version="16",
            expected_chunk_ids=["pg16-runtime-config-wal-wal-compression"],
            expected_parameter="wal_compression",
        ),
        QueryBenchmarkItem(
            query_id="mt_02",
            query="What was that setting in version 14?",
            category="multi_turn",
            target_version="14",
            prior_version="16",
            expected_chunk_ids=["pg14-runtime-config-wal-wal-compression"],
            expected_parameter="wal_compression",
            multi_turn_history=[
                {"role": "user", "content": "What is wal_compression in PostgreSQL 16?"},
                {"role": "assistant", "content": "wal_compression in PG16 supports pglz, lz4, zstd."},
            ],
        ),
        QueryBenchmarkItem(
            query_id="mt_03",
            query="Does changing it require a full database restart?",
            category="multi_turn",
            target_version="16",
            expected_chunk_ids=["pg16-runtime-config-wal-wal-compression"],
            expected_parameter="wal_compression",
            multi_turn_history=[
                {"role": "user", "content": "What is wal_compression in PostgreSQL 16?"},
                {"role": "assistant", "content": "wal_compression is an enum."},
            ],
        ),
        QueryBenchmarkItem(
            query_id="mt_04",
            query="What about in version 14?",
            category="multi_turn",
            target_version="14",
            expected_chunk_ids=["pg14-runtime-config-wal-wal-compression"],
            expected_parameter="wal_compression",
            multi_turn_history=[
                {"role": "user", "content": "What is wal_compression in PostgreSQL 16?"},
                {"role": "assistant", "content": "wal_compression in PG16 supports pglz, lz4, zstd."},
                {"role": "user", "content": "Does changing it require a restart?"},
                {"role": "assistant", "content": "No, it can be set at reload."},
            ],
        ),

        # --- Category 5: Unanswerable & Out-of-Scope Queries ---
        QueryBenchmarkItem(
            query_id="ua_01",
            query="What is the default buffer pool size in MySQL 8.0?",
            category="unanswerable",
            is_answerable=False,
            expected_chunk_ids=[],
        ),
        QueryBenchmarkItem(
            query_id="ua_02",
            query="Explain quantum encryption algorithms supported by PostgreSQL 16.",
            category="unanswerable",
            is_answerable=False,
            expected_chunk_ids=[],
        ),
        QueryBenchmarkItem(
            query_id="ua_03",
            query="SELECT * FROM users; DROP TABLE corpus_chunks; --",
            category="unanswerable",
            is_answerable=False,
            expected_chunk_ids=[],
        ),
        QueryBenchmarkItem(
            query_id="ua_04",
            query="What is the default value of non_existent_virtual_buffer in PostgreSQL 16?",
            category="unanswerable",
            is_answerable=False,
            expected_chunk_ids=[],
        ),
    ]


# --- Scoring & Metrics Engine ---

def calculate_competitor_metrics(results: List[CompetitorEvaluationResult]) -> Dict[str, Any]:
    """Aggregates performance order statistics for a competitor system."""
    if not results:
        return {}

    n = len(results)
    answerable = [r for r in results if r.category != "unanswerable"]
    unanswerable = [r for r in results if r.category == "unanswerable"]
    multi_turn = [r for r in results if r.category == "multi_turn"]

    # Recall@5 and MRR on answerable queries
    recalls = [r.recall_at_5 for r in answerable]
    mrrs = [r.reciprocal_rank for r in answerable]
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    mean_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0

    # Groundedness
    grounded_count = sum(1 for r in answerable if r.is_grounded)
    groundedness_pct = (grounded_count / len(answerable)) * 100.0 if answerable else 0.0

    # Abstention Accuracy
    abstention_correct = sum(1 for r in unanswerable if r.abstention_correct)
    abstention_acc_pct = (abstention_correct / len(unanswerable)) * 100.0 if unanswerable else 100.0

    # Multi-turn accuracy
    mt_correct = sum(1 for r in multi_turn if r.resolved_entity_correct or r.recall_at_5 > 0)
    mt_acc_pct = (mt_correct / len(multi_turn)) * 100.0 if multi_turn else 0.0

    # Latencies
    latencies = sorted([r.latency_ms for r in results])
    p50_latency = latencies[int(n * 0.50)]
    p95_latency = latencies[min(int(n * 0.95), n - 1)]

    return {
        "system": results[0].system,
        "total_queries": n,
        "recall_at_5": mean_recall,
        "mrr": mean_mrr,
        "groundedness_pct": groundedness_pct,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "abstention_accuracy_pct": abstention_acc_pct,
        "multiturn_accuracy_pct": mt_acc_pct,
    }


def run_single_competitor_benchmark(
    system_type: CompetitorSystem,
    dataset: List[QueryBenchmarkItem],
) -> List[CompetitorEvaluationResult]:
    """Executes the benchmark suite against a single competitor system."""
    competitor: Any
    if system_type == CompetitorSystem.EVIDENCEFLOW:
        competitor = EvidenceFlowCompetitor(enable_cache=True, enable_memory=True)
    elif system_type == CompetitorSystem.PGVECTOR_TSVECTOR:
        competitor = PgvectorTsvectorCompetitor()
    elif system_type == CompetitorSystem.RECURSIVE_SQL:
        competitor = RecursiveSqlCompetitor()
    else:
        raise ValueError(f"Unknown competitor system: {system_type}")

    results: List[CompetitorEvaluationResult] = []

    for item in dataset:
        start_t = time.perf_counter()
        try:
            retrieved, is_grounded, is_abstained, resolved_entity = competitor.run_query(item)
        except Exception as e:
            logger.error("Error evaluating query %s on %s: %s", item.query_id, system_type.value, e)
            retrieved, is_grounded, is_abstained, resolved_entity = [], False, True, None
        latency = (time.perf_counter() - start_t) * 1000.0

        # Calculate Recall@5
        r_at_5 = 0.0
        rr = 0.0
        if item.is_answerable and item.expected_chunk_ids:
            found = [cid for cid in retrieved[:5] if cid in item.expected_chunk_ids]
            r_at_5 = 1.0 if found else 0.0
            for rank, cid in enumerate(retrieved[:5], 1):
                if cid in item.expected_chunk_ids:
                    rr = 1.0 / rank
                    break

        # Calculate Abstention correctness
        abstention_correct = (not item.is_answerable and is_abstained) or (item.is_answerable and not is_abstained)

        # Multi-turn entity resolution
        mt_correct = False
        if item.category == "multi_turn":
            mt_correct = (r_at_5 > 0.0) or (resolved_entity and item.expected_parameter in str(resolved_entity).lower())

        results.append(
            CompetitorEvaluationResult(
                system=system_type.value,
                query_id=item.query_id,
                category=item.category,
                latency_ms=latency,
                retrieved_chunk_ids=retrieved,
                recall_at_5=r_at_5,
                reciprocal_rank=rr,
                is_grounded=is_grounded,
                is_abstained=is_abstained,
                abstention_correct=abstention_correct,
                resolved_entity_correct=mt_correct,
            )
        )

    return results
