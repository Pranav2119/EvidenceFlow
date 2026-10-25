"""
audit_wrapper.py: Transparent LangGraph node decorator for Step Ledger & Claim Audit logging.
Wraps execution nodes without modifying their internal logic, measuring execution latency,
token consumption, state transitions, and recording verified claims into PostgreSQL.
"""

import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional
from evidenceflow.state import EvidenceFlowState
from evidenceflow.storage.ledger import record_step, record_claim
from evidenceflow.graph.nodes.verifier import extract_claimed_values, split_answer_into_citation_statements

logger = logging.getLogger(__name__)


def audit_node_wrapper(
    node_fn: Callable[[EvidenceFlowState], Dict[str, Any]],
    node_name: str,
) -> Callable[[EvidenceFlowState], Dict[str, Any]]:
    """
    Transparent decorator for StateGraph nodes.
    Records step execution metrics into step_ledger, and if claims were evaluated, into claim_audit.
    """
    def wrapped_node(state: EvidenceFlowState) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # Resolve thread_id and turn_id from state
        thread_id = getattr(state, "thread_id", "default_thread")
        turn_id = getattr(state, "turn_id", None)
        if not turn_id:
            turn_id = str(uuid.uuid4())

        step_number = len(getattr(state, "audit_trail", [])) + 1

        # Execute the underlying node function
        try:
            result = node_fn(state)
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            try:
                record_step(
                    turn_id=turn_id,
                    thread_id=thread_id,
                    step_number=step_number,
                    node=node_name,
                    action="exception",
                    status="error",
                    decided_by="rule",
                    latency_ms=latency_ms,
                    details={"error": str(e)},
                )
            except Exception as db_err:
                logger.error("Failed to write error step to ledger: %s", db_err)
            raise e

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Determine status and categorical abstention code
        is_abstained = result.get("is_abstained", getattr(state, "is_abstained", False))
        abstention_code = result.get("abstention_code", getattr(state, "abstention_code", None))
        status = "abstained" if is_abstained else "success"

        # Attribute decision origin (rule / llm / fallback)
        if node_name == "answer_writer":
            answer = result.get("draft_answer", "")
            if "According to PostgreSQL" in answer:
                decided_by = "fallback"
            else:
                decided_by = "llm"
        else:
            decided_by = "rule"

        # Extract audit action, token metrics, and details from latest audit entry
        new_audit = result.get("audit_trail", [])
        action = f"{node_name}_executed"
        tokens = {}
        details = {}

        if new_audit and isinstance(new_audit, list):
            latest_entry = new_audit[-1]
            if isinstance(latest_entry, dict):
                action = latest_entry.get("action", action)
                tokens = {
                    k: v for k, v in latest_entry.items()
                    if k in {"prompt_tokens", "candidate_tokens", "total_tokens"}
                }
                details = {
                    k: v for k, v in latest_entry.items()
                    if k not in {"prompt_tokens", "candidate_tokens", "total_tokens", "action", "node"}
                }

        # Write execution row to step_ledger
        try:
            record_step(
                turn_id=turn_id,
                thread_id=thread_id,
                step_number=step_number,
                node=node_name,
                action=action,
                status=status,
                decided_by=decided_by,
                latency_ms=latency_ms,
                abstention_code=abstention_code,
                tokens=tokens,
                details=details,
            )
        except Exception as db_err:
            logger.error("Failed to record step to step_ledger: %s", db_err)

        # If node is verifier, evaluate and record claims into claim_audit
        if node_name == "verifier":
            draft_answer = getattr(state, "draft_answer", "")
            passages = getattr(state, "retrieved_passages", [])
            passage_map = {p.get("chunk_id"): p for p in passages}
            target_version = getattr(state, "target_version", "16") or "16"
            l1_passed = result.get("l1_passed", False)
            verifier_verdict = "PASS" if l1_passed else "FAIL"
            verifier_code = "VERIFIED" if l1_passed else (abstention_code or "UNVERIFIED")

            statements = split_answer_into_citation_statements(draft_answer)
            for stmt_text, clause_cids in statements:
                claimed_values = extract_claimed_values(stmt_text)
                for cid in clause_cids:
                    p = passage_map.get(cid, {})
                    c_sha = p.get("sha256", "sha256_unindexed")
                    c_ver = p.get("version", target_version)
                    try:
                        record_claim(
                            turn_id=turn_id,
                            thread_id=thread_id,
                            statement=stmt_text,
                            chunk_id=cid,
                            chunk_sha256=c_sha,
                            chunk_version=c_ver,
                            verifier_verdict=verifier_verdict,
                            verifier_code=verifier_code,
                            claimed_values=claimed_values,
                        )
                    except Exception as db_err:
                        logger.error("Failed to record claim to claim_audit: %s", db_err)

        # Ensure turn_id propagates across graph state
        if "turn_id" not in result:
            result["turn_id"] = turn_id

        return result

    return wrapped_node
