"""
retriever.py: Grounded retrieval node executing version-filtered vector search in Qdrant
with confidence thresholding, lexical grounding, cross-version detection, and title boosting.
"""

import re
from typing import Any, Dict, List, Optional, Set
from evidenceflow.config import get_settings
from evidenceflow.state import EvidenceFlowState
from evidenceflow.storage.qdrant_client import get_evidence_qdrant_client
from evidenceflow.storage.bm25 import get_bm25_index

COMMON_QUERY_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "does", "have", "from", "with", "postgres", "postgresql", "version", "release",
    "configure", "setting", "parameter", "options", "value", "default", "explain",
    "show", "tell", "about", "describe", "find", "list", "between", "this", "that",
    "the", "and", "for", "are", "can", "should", "would", "could", "into", "also",
}


def normalize_token(token: str) -> str:
    """Normalizes hyphens and underscores for resilient keyword comparison."""
    return re.sub(r"[-_]", "", token.lower())


def extract_distinctive_query_terms(query: str) -> Set[str]:
    """
    Extracts informative keywords from query excluding stopwords and generic phrasing.
    Preserves compound tokens with hyphenation normalized.
    """
    words = re.findall(r"\b[a-z0-9_]{3,}\b", query.lower())
    return {w for w in words if w not in COMMON_QUERY_STOPWORDS and not w.isdigit()}


def compute_reciprocal_rank_fusion(
    query_results: List[List[Dict[str, Any]]],
    exact_targets: Optional[List[str]] = None,
    k: int = 60,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) with exact title/parameter match boosting.
    RRF score = sum(1.0 / (k + rank)) across query result lists.
    Exact-name boost: candidate passages whose title or chunk_id directly defines
    the target parameter receive a +0.50 boost, ensuring exact parameters rank #1.
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

    # Apply exact parameter boost if targets are present
    if exact_targets:
        normalized_targets = [normalize_token(t) for t in exact_targets]
        for cid, passage in passage_by_id.items():
            norm_cid = normalize_token(cid)
            norm_title = normalize_token(passage.get("title", ""))
            for target in normalized_targets:
                if target in norm_title or target in norm_cid:
                    # Give substantial boost to exact identifier chunk
                    rrf_scores[cid] += 0.50
                    break

    # Rank passages by aggregated RRF score descending
    sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    fused_passages = []
    for cid in sorted_cids[:limit]:
        item = dict(passage_by_id[cid])
        item["rrf_score"] = rrf_scores[cid]
        fused_passages.append(item)

    return fused_passages


def check_parameter_in_other_versions(param: str, client, current_version: str) -> Optional[str]:
    """
    Version-aware check: Verifies if a parameter missing in current_version exists
    in other indexed PostgreSQL versions (e.g. 15 or 16).
    """
    for v in ["16", "15", "14"]:
        if v == current_version:
            continue
        hits = client.search(query=param, version=v, limit=2)
        norm_param = normalize_token(param)
        for h in hits:
            if norm_param in normalize_token(h.get("title", "")) or norm_param in normalize_token(h.get("chunk_id", "")):
                return v
    return None


def retriever_node(state: EvidenceFlowState, limit: Optional[int] = None) -> Dict[str, Any]:
    """
    LangGraph node: Queries Qdrant using sub-queries from plan, scoped to target PostgreSQL version.
    Applies Day 4 Retrieval Guardrails:
      1. Raw dense cosine similarity thresholding (never compares RRF scores against threshold).
      2. Strict distinctive keyword grounding check (hyphen-normalized).
      3. Cross-version presence detection (VERSION_MISSING).
      4. Hybrid exact-title boosting so exact parameter chunks rank #1.
    """
    if state.is_abstained:
        return {}

    settings = get_settings()
    retrieval_limit = limit or settings.RETRIEVAL_LIMIT
    threshold = settings.RETRIEVAL_SCORE_THRESHOLD

    query = state.query
    target_version = state.target_version or "16"
    plan = list(state.plan) if state.plan else [query]
    existing_audit = list(state.audit_trail)

    client = get_evidence_qdrant_client()
    bm25_idx = get_bm25_index()

    # Collect result lists for each subquery: dense vector + BM25 lexical
    query_results: List[List[Dict[str, Any]]] = []
    all_raw_passages: List[Dict[str, Any]] = []
    dense_passages: List[Dict[str, Any]] = []

    for subquery in plan:
        dense_results = client.search(query=subquery, version=target_version, limit=retrieval_limit)
        bm25_results = bm25_idx.search(query=subquery, version=target_version, limit=retrieval_limit)

        if dense_results:
            query_results.append(dense_results)
            all_raw_passages.extend(dense_results)
            dense_passages.extend(dense_results)

        if bm25_results:
            query_results.append(bm25_results)
            all_raw_passages.extend(bm25_results)

    # Guardrail Check 1: Empty retrieval results
    if not all_raw_passages:
        fail_reason = f"No documentation passages found for PostgreSQL {target_version}."
        audit_entry = {
            "node": "retriever",
            "action": "abstained",
            "code": "MISSING_KEYWORD",
            "reason": fail_reason,
        }
        return {
            "retrieved_passages": [],
            "is_abstained": True,
            "abstention_code": "MISSING_KEYWORD",
            "abstention_reason": fail_reason,
            "audit_trail": existing_audit + [audit_entry],
        }

    # Guardrail Check 2: Raw vector similarity threshold (dense cosine score, NOT RRF)
    max_dense_score = max((p.get("score", 0.0) for p in dense_passages), default=0.0)
    if max_dense_score < threshold:
        fail_reason = (
            f"Retrieval Guardrail: Highest document similarity score ({max_dense_score:.4f}) is below "
            f"the required confidence threshold ({threshold:.2f})."
        )
        audit_entry = {
            "node": "retriever",
            "action": "abstained_low_confidence",
            "code": "LOW_SCORE",
            "max_score": max_dense_score,
            "threshold": threshold,
            "reason": fail_reason,
        }
        return {
            "retrieved_passages": [],
            "is_abstained": True,
            "abstention_code": "LOW_SCORE",
            "abstention_reason": fail_reason,
            "audit_trail": existing_audit + [audit_entry],
        }

    # Guardrail Check 3: Strict lexical grounding verification
    distinctive_terms = extract_distinctive_query_terms(query)
    combined_normalized_text = " ".join(
        f"{normalize_token(p.get('chunk_id', ''))} {normalize_token(p.get('title', ''))} {normalize_token(p.get('content', ''))}"
        for p in all_raw_passages
    )

    # Check if any specific parameter or distinctive modifier is completely absent
    unmatched_terms = []
    for term in distinctive_terms:
        norm_t = normalize_token(term)
        if norm_t not in combined_normalized_text:
            unmatched_terms.append(term)

    # If all terms or critical compound terms are missing, check if it exists in other versions
    if len(unmatched_terms) == len(distinctive_terms) or any("_" in t for t in unmatched_terms):
        # Extract snake_case parameter name if present (requires underscore to avoid matching generic verbs)
        param_match = re.search(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b", query.lower())
        target_param = param_match.group(0) if param_match else None

        # Version-aware check: check if it exists in another version
        if target_param:
            alt_version = check_parameter_in_other_versions(target_param, client, target_version)
            if alt_version:
                version_reason = (
                    f"Version Scope: The parameter '{target_param}' is present in PostgreSQL {alt_version} "
                    f"documentation, but does not exist in PostgreSQL {target_version}."
                )
                audit_entry = {
                    "node": "retriever",
                    "action": "abstained_version_missing",
                    "code": "VERSION_MISSING",
                    "param": target_param,
                    "target_version": target_version,
                    "alt_version": alt_version,
                    "reason": version_reason,
                }
                return {
                    "retrieved_passages": [],
                    "is_abstained": True,
                    "abstention_code": "VERSION_MISSING",
                    "abstention_reason": version_reason,
                    "audit_trail": existing_audit + [audit_entry],
                }

        # Check for did-you-mean suggestions using difflib against indexed setting names
        suggestions = []
        for term in unmatched_terms:
            sugg = bm25_idx.suggest_setting_name(term)
            if sugg:
                suggestions.append(sugg)

        suggestion_hint = ""
        if suggestions:
            suggestion_hint = f" Did you mean '{suggestions[0]}'?"

        fail_reason = (
            f"Retrieval Guardrail: Query terms {sorted(unmatched_terms)} do not exist in PostgreSQL {target_version} "
            f"documentation (possible non-existent feature or unanswerable query).{suggestion_hint}"
        )
        audit_entry = {
            "node": "retriever",
            "action": "abstained_ungrounded_terms",
            "code": "MISSING_KEYWORD",
            "unmatched_terms": sorted(unmatched_terms),
            "reason": fail_reason,
        }
        return {
            "retrieved_passages": [],
            "is_abstained": True,
            "abstention_code": "MISSING_KEYWORD",
            "abstention_reason": fail_reason,
            "audit_trail": existing_audit + [audit_entry],
        }

    # Extract target parameter names for title-boosting
    param_matches = re.findall(r"\b[a-z][a-z0-9_]{2,}[a-z0-9]\b", query.lower())
    exact_targets = [p for p in param_matches if "_" in p]

    # Fuse rankings across sub-queries using RRF with exact title boosting
    top_passages = compute_reciprocal_rank_fusion(
        query_results,
        exact_targets=exact_targets,
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
        "abstention_code": None,
        "abstention_reason": None,
        "audit_trail": existing_audit + [audit_entry],
    }
