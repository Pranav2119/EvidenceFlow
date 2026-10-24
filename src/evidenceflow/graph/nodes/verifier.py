"""
verifier.py: Deterministic L1 Citation Existence and L1.5 Value-Presence Verifier.
Executes pure Python verification (ZERO LLM calls) to guarantee:
  1. Citation Existence: Every cited chunk ID exists in retrieved context.
  2. Citation Completeness: Ungrounded statements with zero citations are caught.
  3. L1.5 Value Presence: Numeric parameters, units, and values cited in a claim
     actually exist verbatim in the cited passage text.
"""

import re
from typing import Any, Dict, List, Set, Union
from evidenceflow.state import EvidenceFlowState, Citation


def verify_statement_value_presence(
    statement: str,
    passage_content: str,
) -> List[str]:
    """
    L1.5 Check: Verifies that specific numbers and memory/time quantities
    mentioned in the statement exist in the cited passage text.
    Returns list of missing/hallucinated values, if any.
    """
    # Extract numeric quantities (e.g. '200', '9999', '128MB', '8kB', '10s')
    quantities = re.findall(
        r"\b\d+(?:\.\d+)?(?:\s*(?:ms|min|kB|MB|GB|TB|pages|bytes|rows))?\b",
        statement,
        re.IGNORECASE,
    )

    # Filter out common document structure numbers (like footnote index numbers '1', '2', or '16' for PG version)
    filtered_quantities = {
        q.strip().lower()
        for q in quantities
        if q.strip() not in {"1", "2", "3", "4", "5", "14", "15", "16"}
    }

    content_lower = passage_content.lower()
    missing_values = []
    for q in filtered_quantities:
        if q not in content_lower:
            missing_values.append(q)

    return missing_values


def verifier_node(state: EvidenceFlowState) -> Dict[str, Any]:
    """
    LangGraph node: Deterministic citation existence and value-presence check.
    Fails L1 verification if:
      - Any cited chunk is hallucinated (not retrieved)
      - The answer makes claims with zero citations
      - A numeric or configuration value in a claim does not appear in the source passage
    """
    draft_answer = state.draft_answer
    citations = list(state.citations)
    passages = list(state.retrieved_passages)
    is_abstained = state.is_abstained
    abstention_reason = state.abstention_reason
    existing_audit = list(state.audit_trail)

    # If already abstained by upstream guardrails, pass through
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
    passage_map = {p["chunk_id"]: p.get("content", "") for p in passages}
    retrieved_chunk_ids = set(passage_map.keys())

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

    # Failure Mode 3 (L1.5): Value Presence Verification
    # Split text into statements/clauses ending in a citation
    clauses = re.findall(r"([^.\n;]+?\[\^([^\]]+)\])", draft_answer)
    for clause_text, cid in clauses:
        passage_content = passage_map.get(cid, "")
        missing = verify_statement_value_presence(clause_text, passage_content)
        if missing:
            fail_reason = (
                f"L1.5 Value Verification Failed: Cited quantity/value {missing} does not "
                f"appear in source passage '{cid}'."
            )
            audit_entry = {
                "node": "verifier",
                "action": "l1_5_failed_value_divergence",
                "chunk_id": cid,
                "missing_values": missing,
                "l1_passed": False,
            }
            return {
                "l1_passed": False,
                "is_abstained": True,
                "abstention_reason": fail_reason,
                "audit_trail": existing_audit + [audit_entry],
            }

    # Mark Citations as verified
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
