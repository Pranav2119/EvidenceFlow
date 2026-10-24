"""
retriever.py: Grounded retrieval node executing version-filtered vector search in Qdrant.
"""

from typing import Any, Dict, List
from evidenceflow.state import EvidenceFlowState
from evidenceflow.storage.qdrant_client import EvidenceQdrantClient


def retriever_node(state: EvidenceFlowState, limit: int = 5) -> Dict[str, Any]:
    """
    LangGraph node: Queries Qdrant using the sub-queries from the plan,
    scoped strictly to the target PostgreSQL version via filterable HNSW.
    """
    query = state.query if hasattr(state, "query") else state["query"]
    target_version = state.target_version if hasattr(state, "target_version") else state.get("target_version")
    plan = list(state.plan if hasattr(state, "plan") else state.get("plan", []))
    existing_audit = list(state.audit_trail if hasattr(state, "audit_trail") else state.get("audit_trail", []))

    if not plan:
        plan = [query]

    client = EvidenceQdrantClient()
    seen_ids = set()
    aggregated_passages: List[Dict[str, Any]] = []

    for subquery in plan:
        results = client.search(query=subquery, version=target_version, limit=limit)
        for passage in results:
            cid = passage["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                aggregated_passages.append(passage)

    # Sort aggregated passages by score descending and truncate to limit
    aggregated_passages.sort(key=lambda p: p.get("score", 0.0), reverse=True)
    top_passages = aggregated_passages[:limit]

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
