"""
memory_resolver.py: LangGraph node resolving multi-turn follow-ups against Neo4j memory.
Enforces [Req 4, 5]:
  - Explicit settings win over memory.
  - Refusals and multi-entity turns trigger clarifying questions.
  - Memory ONLY rewrites the query for downstream pipeline execution.
"""

import logging
from typing import Any, Dict
from evidenceflow.state import EvidenceFlowState
from evidenceflow.memory.graph_client import resolve_entity_for_turn
from evidenceflow.guardrails.input_guardrail import check_pre_resolution_guards

logger = logging.getLogger(__name__)


def memory_resolver_node(state: EvidenceFlowState) -> Dict[str, Any]:
    """
    Inspects active conversational thread memory in Neo4j.
    Enforces pre-resolution prompt injection and explicit off-domain checks
    on the ORIGINAL query text before any memory rewriting occurs [Rule c].
    If the query is a follow-up pronoun query, rewrites it into a self-contained question.
    If ambiguous or referencing a refusal, routes to clarification abstention.
    If explicit or not a follow-up, passes through with 0 modification.
    """
    thread_id = state.thread_id or "default_thread"
    query = state.query

    # Rule c: Pre-flight injection and explicit off-domain check on ORIGINAL text
    is_safe, error_code, fail_reason = check_pre_resolution_guards(query)
    if not is_safe:
        logger.warning(
            "Pre-resolution guard stopped query on thread %s: code=%s reason=%s",
            thread_id,
            error_code,
            fail_reason,
        )
        existing_audit = list(state.audit_trail)
        audit_entry = {
            "node": "memory_resolver",
            "action": "pre_resolution_abstained",
            "code": error_code,
            "reason": fail_reason,
            "query": query,
        }
        return {
            "is_abstained": True,
            "abstention_code": error_code,
            "abstention_reason": fail_reason,
            "draft_answer": f"I cannot provide an answer for this question.\n\n- **Status**: ABSTAINED ({error_code})\n- **Reason**: {fail_reason}\n\n*EvidenceFlow AI adheres to strict groundedness: answers are only generated when verifiable evidence exists in official documentation.*",
            "audit_trail": existing_audit + [audit_entry],
        }

    resolution = resolve_entity_for_turn(thread_id, query)

    if resolution["status"] == "resolved":
        logger.info(
            "Memory resolved follow-up for thread %s: entity='%s', version='%s' (took %.2f ms)",
            thread_id,
            resolution.get("entity_name"),
            resolution.get("target_version"),
            resolution.get("latency_ms", 0.0),
        )
        return {
            "query": resolution["rewritten_query"],
            "target_version": resolution["target_version"],
            "prior_version": resolution.get("prior_version"),
        }

    elif resolution["status"] == "ambiguous":
        logger.info(
            "Memory flagged ambiguous follow-up for thread %s: %s (took %.2f ms)",
            thread_id,
            resolution.get("reason"),
            resolution.get("latency_ms", 0.0),
        )
        clarification_msg = resolution.get("clarification", "Could you clarify which parameter you are referring to?")
        return {
            "is_abstained": True,
            "abstention_code": "AMBIGUOUS_FOLLOWUP",
            "abstention_reason": clarification_msg,
            "draft_answer": clarification_msg,
        }

    # 'passthrough' status: carry target_version if turn 2 lacks one
    diff = {}
    if not state.target_version and resolution.get("target_version"):
        diff["target_version"] = resolution["target_version"]
    return diff
