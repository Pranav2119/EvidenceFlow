"""
planner.py: Query decomposition and version scope resolution node.
Identifies target PostgreSQL version and formulates retrieval sub-queries.
"""

import re
from typing import Any, Dict, List, Optional
from evidenceflow.state import EvidenceFlowState


def extract_target_version(query: str, current_version: Optional[str] = None) -> Optional[str]:
    """
    Extracts explicit PostgreSQL version from query (e.g. 'Postgres 16', 'v15', 'PostgreSQL 14').
    If none found in query, preserves current_version or defaults to '16'.
    """
    if current_version:
        return str(current_version)

    # Regex matching "postgres 16", "postgresql 15", "pg14", "v16", "version 15"
    match = re.search(r"\b(?:postgres(?:ql)?|pg|v|version)?\s*(14|15|16)\b", query, re.IGNORECASE)
    if match:
        return match.group(1)

    return "16"  # Default to latest indexed version


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
    LangGraph node: Analyzes user query, infers target PostgreSQL version,
    and formulates discrete retrieval sub-queries.
    """
    query = state.query if hasattr(state, "query") else state["query"]
    existing_version = state.target_version if hasattr(state, "target_version") else state.get("target_version")
    existing_audit = list(state.audit_trail if hasattr(state, "audit_trail") else state.get("audit_trail", []))

    target_version = extract_target_version(query, existing_version)
    plan = generate_subqueries(query)

    audit_entry = {
        "node": "planner",
        "action": "plan_generated",
        "target_version": target_version,
        "plan": plan,
    }

    return {
        "target_version": target_version,
        "plan": plan,
        "audit_trail": existing_audit + [audit_entry],
    }
