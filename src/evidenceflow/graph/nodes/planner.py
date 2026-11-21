"""
planner.py: Query decomposition and version scope resolution node.
Identifies target PostgreSQL version and formulates retrieval sub-queries.
Detects cross-version comparative queries to populate prior_version for comparative retrieval.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from evidenceflow.state import EvidenceFlowState


def extract_versions(query: str, current_version: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Extracts target and prior versions from query.
    If multiple versions are mentioned (e.g. 'Postgres 14 vs 16' or 'between 14 and 16'),
    returns (newer_version, older_version) to enable cross-version comparative retrieval.
    """
    matches = re.findall(r"\b(?:postgres(?:ql)?|pg|v|version|release)?\s*(14|15|16)\b", query, re.IGNORECASE)
    cleaned = []
    for m in matches:
        if m in {"14", "15", "16"} and m not in cleaned:
            cleaned.append(m)

    if len(cleaned) >= 2:
        sorted_v = sorted(cleaned, key=lambda x: int(x))
        return sorted_v[-1], sorted_v[0]
    elif len(cleaned) == 1:
        return cleaned[0], None

    return str(current_version) if current_version else "16", None


def extract_target_version(query: str, current_version: Optional[str] = None) -> str:
    """Backward compatible wrapper returning only the target version."""
    target_v, _ = extract_versions(query, current_version)
    return target_v


def generate_subqueries(query: str) -> List[str]:
    """
    Extracts targeted sub-queries and parameter names from user question.
    Preserves exact parameter names (e.g. 'vacuum_cost_limit') for high-precision retrieval.
    """
    subqueries = [query.strip()]

    # Extract snake_case GUC identifiers (e.g. shared_buffers, max_wal_size, autovacuum_naptime)
    param_matches = re.findall(r"\b[a-z][a-z0-9_]{2,}[a-z0-9]\b", query.lower())
    common_stopwords = {"what", "when", "where", "which", "does", "have", "from", "with", "postgres", "postgresql", "version"}
    significant_params = [p for p in param_matches if "_" in p and p not in common_stopwords]

    for param in significant_params:
        if param not in subqueries:
            subqueries.append(param)

    return subqueries


def planner_node(state: EvidenceFlowState) -> Dict[str, Any]:
    """
    LangGraph node: Analyzes user query, infers target and prior PostgreSQL versions,
    and formulates discrete retrieval sub-queries.
    """
    query = state.query
    existing_version = state.target_version
    existing_audit = list(state.audit_trail)

    target_version, detected_prior_version = extract_versions(query, existing_version)
    prior_version = state.prior_version or detected_prior_version
    plan = generate_subqueries(query)

    audit_entry = {
        "node": "planner",
        "action": "plan_generated",
        "target_version": target_version,
        "prior_version": prior_version,
        "plan": plan,
    }

    return {
        "target_version": target_version,
        "prior_version": prior_version,
        "plan": plan,
        "audit_trail": existing_audit + [audit_entry],
    }
