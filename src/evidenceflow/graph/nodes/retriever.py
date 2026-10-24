"""
retriever.py: Grounded retrieval node executing version-filtered vector search in Qdrant.
"""

from typing import Any, Dict, List, Optional
from evidenceflow.config import get_settings
from evidenceflow.state import EvidenceFlowState
from evidenceflow.storage.qdrant_client import get_evidence_qdrant_client


def compute_reciprocal_rank_fusion(
    query_results: List[List[Dict[str, Any]]],
    k: int = 60,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    D3-8 fix: Reciprocal Rank Fusion (RRF) to merge ranked lists from multiple sub-queries.
    RRF score = sum(1.0 / (k + rank)) across query result lists.
    Eliminates cross-query cosine score incomparability.
    """
    rrf_scores: Dict[str, float] = {}
    passage_by_id: Dict[str, Dict[str, Any]] = {}

    for results in query_results:
        for rank, passage in enumerate(results, start=1):
            cid = passage["chunk_id"]
            score_delta = 1.0 / (k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_delta
            if cid not in passage_by_id:
                passage_by_id[cid] = passage

    # Rank passages by aggregated RRF score descending
    sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    fused_passages = []
    for cid in sorted_cids[:limit]:
        item = dict(passage_by_id[cid])
        item["rrf_score"] = rrf_scores[cid]
        fused_passages.append(item)

    return fused_passages


def retriever_node(state: EvidenceFlowState, limit: Optional[int] = None) -> Dict[str, Any]:
    """
    LangGraph node: Queries Qdrant using the sub-queries from the plan,
    scoped strictly to the target PostgreSQL version via filterable HNSW.
    D3-2: Direct Pydantic attribute access.
    D3-3: Uses shared EvidenceQdrantClient singleton.
    D3-8: Merges multi-query result sets with Reciprocal Rank Fusion.
    D3-14: Configurable retrieval limit via Settings.
    """
    settings = get_settings()
    retrieval_limit = limit or settings.RETRIEVAL_LIMIT

    query = state.query
    target_version = state.target_version
    plan = list(state.plan) if state.plan else [query]
    existing_audit = list(state.audit_trail)

    # D3-3: Shared client singleton — connection pool reused across invocations
    client = get_evidence_qdrant_client()

    # Collect result lists for each subquery
    query_results: List[List[Dict[str, Any]]] = []
    for subquery in plan:
        results = client.search(query=subquery, version=target_version, limit=retrieval_limit)
        if results:
            query_results.append(results)

    # D3-8: Fuse rankings across sub-queries using RRF
    top_passages = compute_reciprocal_rank_fusion(
        query_results,
        k=60,
        limit=retrieval_limit,
    )

    audit_entry = {
        "node": "retriever",
        "action": "passages_retrieved",
        "target_version": target_version,
        "count": len(top_passages),
        "chunk_ids": [p["chunk_id"] for p in top_passages],
    }

    return {
        "retrieved_passages": top_passages,
        "audit_trail": existing_audit + [audit_entry],
    }
