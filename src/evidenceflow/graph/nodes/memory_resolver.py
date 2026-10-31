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

logger = logging.getLogger(__name__)


def memory_resolver_node(state: EvidenceFlowState) -> Dict[str, Any]:
    """
    Inspects active conversational thread memory in Neo4j.
    If the query is a follow-up pronoun query, rewrites it into a self-contained question.
    If ambiguous or referencing a refusal, routes to clarification abstention.
    If explicit or not a follow-up, passes through with 0 modification.
    """
    thread_id = state.thread_id or "default_thread"
    query = state.query

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

    # 'passthrough' status: return empty diff, preserving state
    return {}
