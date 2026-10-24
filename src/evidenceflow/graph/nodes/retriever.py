"""
retriever.py: Grounded retrieval node executing version-filtered vector search in Qdrant
with confidence thresholding and lexical grounding guardrails.
"""

import re
from typing import Any, Dict, List, Optional, Set
from evidenceflow.config import get_settings
from evidenceflow.state import EvidenceFlowState
from evidenceflow.storage.qdrant_client import get_evidence_qdrant_client

COMMON_QUERY_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "does", "have", "from", "with", "postgres", "postgresql", "version", "release",
    "configure", "setting", "parameter", "options", "value", "default", "explain",
    "show", "tell", "about", "describe", "find", "list", "between", "this", "that",
    "the", "and", "for", "are", "can", "should", "would", "could", "into", "also",
}


def extract_distinctive_query_terms(query: str) -> Set[str]:
    """
    Extracts informative keywords from query excluding stopwords and generic phrasing.
    """
    words = re.findall(r"\b[a-z0-9_]{3,}\b", query.lower())
    return {w for w in words if w not in COMMON_QUERY_STOPWORDS and not w.isdigit()}


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
    LangGraph node: Queries Qdrant using sub-queries from the plan, scoped to target PostgreSQL version.
    Applies Day 4 Retrieval Guardrails:
      1. Score threshold filter
      2. Lexical grounding check on distinctive query terms
    """
    # If upstream input guardrail already abstained, pass through immediately
    if state.is_abstained:
        return {}

    settings = get_settings()
    retrieval_limit = limit or settings.RETRIEVAL_LIMIT
    threshold = settings.RETRIEVAL_SCORE_THRESHOLD

    query = state.query
    target_version = state.target_version
    plan = list(state.plan) if state.plan else [query]
    existing_audit = list(state.audit_trail)

    client = get_evidence_qdrant_client()

    # Collect result lists for each subquery
    query_results: List[List[Dict[str, Any]]] = []
    all_raw_passages: List[Dict[str, Any]] = []

    for subquery in plan:
        results = client.search(query=subquery, version=target_version, limit=retrieval_limit)
        if results:
            query_results.append(results)
            all_raw_passages.extend(results)

    # Guardrail Check 1: Empty retrieval results
    if not all_raw_passages:
        fail_reason = f"Retrieval Guardrail: No documentation passages found for PostgreSQL {target_version}."
        audit_entry = {
            "node": "retriever",
            "action": "abstained",
            "reason": fail_reason,
        }
        return {
            "retrieved_passages": [],
            "is_abstained": True,
            "abstention_reason": fail_reason,
            "audit_trail": existing_audit + [audit_entry],
        }

    # Guardrail Check 2: Raw vector similarity threshold
    max_raw_score = max(p.get("score", 0.0) for p in all_raw_passages)
    if max_raw_score < threshold:
        fail_reason = (
            f"Retrieval Guardrail: Document similarity score ({max_raw_score:.4f}) is below "
            f"the required confidence threshold ({threshold:.2f})."
        )
        audit_entry = {
            "node": "retriever",
            "action": "abstained_low_confidence",
            "max_score": max_raw_score,
            "threshold": threshold,
            "reason": fail_reason,
        }
        return {
            "retrieved_passages": [],
            "is_abstained": True,
            "abstention_reason": fail_reason,
            "audit_trail": existing_audit + [audit_entry],
        }

    # Guardrail Check 3: Lexical grounding verification
    # Ensures at least one distinctive query term is actually mentioned in the retrieved evidence
    distinctive_terms = extract_distinctive_query_terms(query)
    if distinctive_terms:
        combined_text = " ".join(
            f"{p.get('chunk_id', '')} {p.get('title', '')} {p.get('content', '')}".lower()
            for p in all_raw_passages
        )
        matched_terms = {t for t in distinctive_terms if t in combined_text}
        if not matched_terms:
            unmatched = sorted(list(distinctive_terms))
            fail_reason = (
                f"Retrieval Guardrail: Query terms {unmatched} do not exist in PostgreSQL {target_version} "
                f"documentation (possible non-existent feature or unanswerable query)."
            )
            audit_entry = {
                "node": "retriever",
                "action": "abstained_ungrounded_terms",
                "unmatched_terms": unmatched,
                "reason": fail_reason,
            }
            return {
                "retrieved_passages": [],
                "is_abstained": True,
                "abstention_reason": fail_reason,
                "audit_trail": existing_audit + [audit_entry],
            }

    # Fuse rankings across sub-queries using RRF
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
        "is_abstained": False,
        "abstention_reason": None,
        "audit_trail": existing_audit + [audit_entry],
    }
