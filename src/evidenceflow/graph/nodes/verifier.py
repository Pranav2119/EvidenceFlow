"""
verifier.py: Deterministic L1 Citation Existence and Provenance Verifier.
Executes pure Python verification (ZERO LLM calls) to guarantee that every
cited chunk exists in retrieved evidence and that claims do not hallucinate citations.
"""

import re
from typing import Any, Dict, List, Union
from evidenceflow.state import EvidenceFlowState, Citation


def verifier_node(state: EvidenceFlowState) -> Dict[str, Any]:
    """
    LangGraph node: Deterministic citation existence and provenance check.
    Verifies that every [^chunk_id] marker in the draft answer references a chunk
    that was actually retrieved in context. Prevents citation hallucinations.
    (Note: Statement-level value presence and NLI entailment are handled in Day 4 and Tier 3).
    D3-2 fix: Direct Pydantic attribute access.
    D3-4 fix: Handles strongly-typed Citation models.
    D3-6 fix: Accurate docstring scope.
    """
    draft_answer = state.draft_answer
    citations = list(state.citations)
    passages = list(state.retrieved_passages)
    is_abstained = state.is_abstained
    abstention_reason = state.abstention_reason
    existing_audit = list(state.audit_trail)

    # If already abstained by planner or answer_writer, pass through
    if is_abstained:
        audit_entry = {
            "node": "verifier",
            "action": "passed_through_abstention",
            "l1_passed": True,
            "reason": abstention_reason,
        }
        return {
            "l1_passed": True,
            "is_abstained": True,
            "abstention_reason": abstention_reason,
            "audit_trail": existing_audit + [audit_entry],
        }

    # Extract all citation markers [^chunk_id] in the draft answer
    cited_chunk_ids = re.findall(r"\[\^([^\]]+)\]", draft_answer)
    retrieved_chunk_ids = {p["chunk_id"] for p in passages}

    # Failure Mode 1: Answer has no citations at all
    if not cited_chunk_ids:
        fail_reason = "L1 Failed: Answer contains ungrounded statements with zero citations."
        audit_entry = {
            "node": "verifier",
            "action": "l1_failed_no_citations",
            "l1_passed": False,
        }
        return {
            "l1_passed": False,
            "is_abstained": True,
            "abstention_reason": fail_reason,
            "audit_trail": existing_audit + [audit_entry],
        }

    # Failure Mode 2: Hallucinated citations (cited chunk was NOT in retrieved_passages)
    hallucinated_ids = [cid for cid in cited_chunk_ids if cid not in retrieved_chunk_ids]
    if hallucinated_ids:
        fail_reason = f"L1 Failed: Hallucinated citation(s) not in retrieved evidence: {hallucinated_ids}"
        audit_entry = {
            "node": "verifier",
            "action": "l1_failed_hallucinated_citations",
            "hallucinated": hallucinated_ids,
            "l1_passed": False,
        }
        return {
            "l1_passed": False,
            "is_abstained": True,
            "abstention_reason": fail_reason,
            "audit_trail": existing_audit + [audit_entry],
        }

    # D3-4: Mark Citation models or dicts as verified
    updated_citations: List[Union[Citation, Dict[str, Any]]] = []
    for c in citations:
        if isinstance(c, Citation):
            is_valid = c.chunk_id in retrieved_chunk_ids
            updated_citations.append(c.model_copy(update={"is_verified": is_valid}))
        else:
            c_copy = dict(c)
            if c_copy.get("chunk_id") in retrieved_chunk_ids:
                c_copy["is_verified"] = True
            updated_citations.append(c_copy)

    audit_entry = {
        "node": "verifier",
        "action": "l1_passed",
        "l1_passed": True,
        "verified_citations_count": len(updated_citations),
    }

    return {
        "l1_passed": True,
        "is_abstained": False,
        "citations": updated_citations,
        "audit_trail": existing_audit + [audit_entry],
    }
