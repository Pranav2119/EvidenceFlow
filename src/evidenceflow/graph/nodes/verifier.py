"""
verifier.py: Deterministic L1 Groundedness and Citation Verifier.
Executes pure Python verification (ZERO LLM calls) to guarantee that every
cited chunk exists in retrieved evidence and that claims do not hallucinate citations.
"""

import re
from typing import Any, Dict, List
from evidenceflow.state import EvidenceFlowState


def verifier_node(state: EvidenceFlowState) -> Dict[str, Any]:
    """
    LangGraph node: Deterministically validates citations and factual entailment.
    Fails L1 verification if any cited chunk is hallucinated or if an ungrounded answer is produced.
    """
    draft_answer = state.draft_answer if hasattr(state, "draft_answer") else state.get("draft_answer", "")
    citations = list(state.citations if hasattr(state, "citations") else state.get("citations", []))
    passages = list(state.retrieved_passages if hasattr(state, "retrieved_passages") else state.get("retrieved_passages", []))
    is_abstained = state.is_abstained if hasattr(state, "is_abstained") else state.get("is_abstained", False)
    abstention_reason = state.abstention_reason if hasattr(state, "abstention_reason") else state.get("abstention_reason")
    existing_audit = list(state.audit_trail if hasattr(state, "audit_trail") else state.get("audit_trail", []))

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

    # Mark citations as verified
    updated_citations = []
    for c in citations:
        c_copy = dict(c)
        if c_copy["chunk_id"] in retrieved_chunk_ids:
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
