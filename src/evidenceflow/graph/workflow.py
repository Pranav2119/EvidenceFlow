"""
workflow.py: EvidenceFlow LangGraph State Machine assembly and compilation.
Wires START -> [optional memory_resolver] -> input_guardrail -> [optional cache_gatekeeper] -> planner -> retriever -> contradiction_analyst -> answer_writer -> verifier -> END
with conditional branching to cache_gatekeeper (bypassing planner/retriever/LLM on deterministic hit)
and abstention_node for security, domain, ambiguity, and grounding failures.
Integrates PostgreSQL checkpointer for process-restart survival, audit wrapper for immutable ledgers,
Neo4j conversational memory graph with claim supercedence/contradiction reasoning, and 5-layer deterministic scoped cache.
"""

import argparse
import logging
import uuid
from typing import Any, Dict, Optional, Tuple
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START, END

from evidenceflow.state import EvidenceFlowState, Citation
from evidenceflow.guardrails.input_guardrail import input_guardrail_node
from evidenceflow.graph.nodes import (
    planner_node,
    retriever_node,
    contradiction_analyst_node,
    l2_judge_node,
    should_invoke_l2_judge,
    answer_writer_node,
    verifier_node,
    abstention_node,
    memory_resolver_node,
    cache_gatekeeper_node,
    get_cache_manager,
)
from evidenceflow.cache.models import CachedCitation
from evidenceflow.graph.audit_wrapper import audit_node_wrapper
from evidenceflow.storage.ledger import (
    get_postgres_connection_string,
    record_turn as pg_record_turn,
    record_step,
    get_next_step_number,
)
from evidenceflow.memory.graph_client import (
    record_turn as neo4j_record_turn,
    extract_explicit_setting,
    extract_all_explicit_settings,
)

logger = logging.getLogger(__name__)

# Cached compiled graph singletons keyed by (enable_memory, enable_cache)
_compiled_apps: Dict[Tuple[bool, bool], Any] = {}


# --- Conditional Routing Functions ---

def route_after_memory_resolver(state: EvidenceFlowState) -> str:
    """Routes to abstention if memory resolver flagged ambiguity or refusal; otherwise proceeds to input guardrail."""
    if state.is_abstained:
        return "abstention"
    return "input_guardrail"


def route_after_input_guardrail(state: EvidenceFlowState) -> str:
    """Routes to abstention if input guardrail detected injection or off-domain query; otherwise cache gatekeeper."""
    if state.is_abstained:
        return "abstention"
    return "cache_gatekeeper"


def route_after_input_guardrail_no_cache(state: EvidenceFlowState) -> str:
    """Routes directly to planner when cache is disabled."""
    if state.is_abstained:
        return "abstention"
    return "planner"


def route_after_cache_gatekeeper(state: EvidenceFlowState) -> str:
    """Routes directly to END if cache hit was deterministically validated; otherwise proceeds to planner."""
    if state.cache_hit:
        return END
    return "planner"


def route_after_retriever(state: EvidenceFlowState) -> str:
    """Routes to abstention if retriever found no evidence or failed score/keyword threshold; otherwise contradiction analyst."""
    if state.is_abstained:
        return "abstention"
    return "contradiction_analyst"


def route_after_verifier(state: EvidenceFlowState) -> str:
    """Routes to abstention if L1 citation check or L1.5 value presence check failed.
    If passed, checks if selective L2 Judge should be invoked for contested/cross-version claims."""
    if not state.l1_passed or state.is_abstained:
        return "abstention"
    if should_invoke_l2_judge(state):
        return "judge"
    return END


def route_after_judge(state: EvidenceFlowState) -> str:
    """Routes to abstention if L2 Judge rejected the answer; otherwise END."""
    if not state.l2_passed or state.is_abstained:
        return "abstention"
    return END


def create_evidenceflow_graph(
    checkpointer: Optional[Any] = None,
    enable_audit: bool = True,
    enable_memory: bool = False,
    enable_cache: bool = True,
):
    """
    Constructs and compiles the EvidenceFlow StateGraph with conditional abstention routes,
    optional transparent audit ledger wrappers, optional memory resolver, optional cache gatekeeper,
    contradiction analyst for cross-version reasoning, and optional persistent checkpointer.
    """
    builder = StateGraph(EvidenceFlowState)

    # 1. Register Nodes
    if enable_audit:
        if enable_memory:
            builder.add_node("memory_resolver", audit_node_wrapper(memory_resolver_node, "memory_resolver"))
        builder.add_node("input_guardrail", audit_node_wrapper(input_guardrail_node, "input_guardrail"))
        if enable_cache:
            builder.add_node("cache_gatekeeper", audit_node_wrapper(cache_gatekeeper_node, "cache_gatekeeper"))
        builder.add_node("planner", audit_node_wrapper(planner_node, "planner"))
        builder.add_node("retriever", audit_node_wrapper(retriever_node, "retriever"))
        builder.add_node("contradiction_analyst", audit_node_wrapper(contradiction_analyst_node, "contradiction_analyst"))
        builder.add_node("answer_writer", audit_node_wrapper(answer_writer_node, "answer_writer"))
        builder.add_node("verifier", audit_node_wrapper(verifier_node, "verifier"))
        builder.add_node("judge", audit_node_wrapper(l2_judge_node, "judge"))
        builder.add_node("abstention", audit_node_wrapper(abstention_node, "abstention"))
    else:
        if enable_memory:
            builder.add_node("memory_resolver", memory_resolver_node)
        builder.add_node("input_guardrail", input_guardrail_node)
        if enable_cache:
            builder.add_node("cache_gatekeeper", cache_gatekeeper_node)
        builder.add_node("planner", planner_node)
        builder.add_node("retriever", retriever_node)
        builder.add_node("contradiction_analyst", contradiction_analyst_node)
        builder.add_node("answer_writer", answer_writer_node)
        builder.add_node("verifier", verifier_node)
        builder.add_node("judge", l2_judge_node)
        builder.add_node("abstention", abstention_node)

    # 2. Register Edges with Conditional Branches
    if enable_memory:
        builder.add_edge(START, "memory_resolver")
        builder.add_conditional_edges(
            "memory_resolver",
            route_after_memory_resolver,
            {"abstention": "abstention", "input_guardrail": "input_guardrail"},
        )
    else:
        builder.add_edge(START, "input_guardrail")

    if enable_cache:
        builder.add_conditional_edges(
            "input_guardrail",
            route_after_input_guardrail,
            {"abstention": "abstention", "cache_gatekeeper": "cache_gatekeeper"},
        )
        builder.add_conditional_edges(
            "cache_gatekeeper",
            route_after_cache_gatekeeper,
            {"planner": "planner", END: END},
        )
    else:
        builder.add_conditional_edges(
            "input_guardrail",
            route_after_input_guardrail_no_cache,
            {"abstention": "abstention", "planner": "planner"},
        )

    builder.add_edge("planner", "retriever")
    builder.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {"abstention": "abstention", "contradiction_analyst": "contradiction_analyst"},
    )
    builder.add_edge("contradiction_analyst", "answer_writer")
    builder.add_edge("answer_writer", "verifier")
    builder.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"abstention": "abstention", "judge": "judge", END: END},
    )
    builder.add_conditional_edges(
        "judge",
        route_after_judge,
        {"abstention": "abstention", END: END},
    )
    builder.add_edge("abstention", END)

    # 3. Compile Graph with Checkpointer
    return builder.compile(checkpointer=checkpointer)


def get_compiled_graph(enable_memory: bool = False, enable_cache: bool = True):
    """
    Returns the cached compiled StateGraph application instance (in-memory).
    """
    global _compiled_apps
    key = (enable_memory, enable_cache)
    if key not in _compiled_apps:
        _compiled_apps[key] = create_evidenceflow_graph(
            enable_audit=True,
            enable_memory=enable_memory,
            enable_cache=enable_cache,
        )
    return _compiled_apps[key]


def run_pipeline(
    query: str,
    version: Optional[str] = None,
    thread_id: str = "default_thread",
    checkpointer: Optional[Any] = None,
    use_postgres_checkpointer: bool = False,
    turn_id: Optional[str] = None,
    turn_number: int = 1,
    enable_memory: bool = False,
    enable_cache: bool = True,
) -> EvidenceFlowState:
    """
    Executes a single pass through the LangGraph workflow, persisting thread checkpoints
    and audit ledger rows into PostgreSQL, Neo4j memory, and 5-layer scoped cache.
    """
    turn_uuid = turn_id or str(uuid.uuid4())
    conn_to_close = None

    if checkpointer is None and use_postgres_checkpointer:
        conn_str = get_postgres_connection_string()
        conn_to_close = psycopg.connect(conn_str, autocommit=True)
        checkpointer = PostgresSaver(conn_to_close)

    if checkpointer is not None:
        app = create_evidenceflow_graph(
            checkpointer=checkpointer,
            enable_audit=True,
            enable_memory=enable_memory,
            enable_cache=enable_cache,
        )
    else:
        app = get_compiled_graph(enable_memory=enable_memory, enable_cache=enable_cache)

    initial_input = {
        "query": query,
        "thread_id": thread_id,
        "turn_id": turn_uuid,
        "turn_number": turn_number,
    }
    if version:
        initial_input["target_version"] = version

    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(initial_input, config=config)

    state = EvidenceFlowState(**result) if isinstance(result, dict) else result
    target_ver = state.target_version or (version or "16")

    # If pipeline succeeded from fresh retrieval/LLM generation, store verified response into L1 Exact cache
    if enable_cache and not state.cache_hit and not state.is_abstained and state.l1_passed and state.draft_answer:
        try:
            cm = get_cache_manager()
            cached_citations = []
            for c in state.citations:
                cid = getattr(c, "chunk_id", None) or (c.get("chunk_id") if isinstance(c, dict) else None)
                chash = getattr(c, "chunk_hash", None)
                if not chash:
                    for p in state.retrieved_passages:
                        if (p.get("chunk_id") or p.get("id")) == cid:
                            chash = p.get("sha256") or p.get("chunk_hash")
                            break
                if cid:
                    cached_citations.append(
                        CachedCitation(
                            passage_id=cid,
                            chunk_hash=chash or "",
                            source_url=getattr(c, "doc_name", "") or "",
                            title=getattr(c, "section", "") or "",
                        )
                    )
            if cached_citations:
                cm.set_exact(
                    query=query,
                    answer=state.draft_answer,
                    citations=cached_citations,
                    target_versions=target_ver,
                )
                logger.info("Saved verified answer to L1 Exact cache for query: '%s' (v%s)", query[:50], target_ver)
        except Exception as e:
            logger.warning("Failed to store verified response into cache: %s", e)

    # 1. Record turn in PostgreSQL turns table
    try:
        pg_record_turn(
            turn_id=turn_uuid,
            thread_id=thread_id,
            turn_number=turn_number,
            target_version=target_ver,
            query=query,
            draft_answer=state.draft_answer,
            is_abstained=state.is_abstained,
            abstention_code=state.abstention_code,
            abstention_reason=state.abstention_reason,
        )
    except Exception as e:
        logger.error("Failed to record turn to PostgreSQL turns table: %s", e)

    # 2. Record turn in Neo4j conversational memory graph
    memory_write_status = "disabled"
    memory_error_msg = None
    if enable_memory:
        try:
            detected_settings = extract_all_explicit_settings(query) or extract_all_explicit_settings(state.query)
            entities = [{"name": s, "type": "ConfigParam"} for s in detected_settings]
            claims = []
            for c in state.citations:
                cid = getattr(c, "chunk_id", None) if not isinstance(c, dict) else c.get("chunk_id")
                quote = getattr(c, "quote", "") if not isinstance(c, dict) else c.get("quote", "")
                is_ver = getattr(c, "is_verified", False) if not isinstance(c, dict) else c.get("is_verified", False)
                if cid:
                    claims.append({
                        "text": quote[:200] if quote else state.draft_answer[:200],
                        "verdict": "VERIFIED" if is_ver else "UNVERIFIED",
                        "passage_id": cid,
                    })

            neo4j_record_turn(
                turn_id=turn_uuid,
                thread_id=thread_id,
                turn_number=turn_number,
                question=query,
                version_scope=target_ver,
                entities=entities,
                claims=claims,
                is_abstained=state.is_abstained,
                abstention_code=state.abstention_code,
                abstention_reason=state.abstention_reason,
            )
            memory_write_status = "success"
        except Exception as e:
            memory_write_status = "error"
            memory_error_msg = str(e)
            logger.error("Failed to record turn to Neo4j conversational memory: %s", e, exc_info=True)

        try:
            db_steps = get_next_step_number(thread_id)
            state_steps = len(getattr(state, "audit_trail", [])) + 1
            step_number = max(state_steps, db_steps)
            record_step(
                turn_id=turn_uuid,
                thread_id=thread_id,
                step_number=step_number,
                node="memory_writer",
                action="record_memory_turn",
                status="success" if memory_write_status == "success" else "error",
                decided_by="rule",
                latency_ms=0.0,
                details={
                    "memory_write_status": memory_write_status,
                    "error": memory_error_msg,
                },
            )
        except Exception as step_err:
            logger.error("Failed to record memory write step to step_ledger: %s", step_err, exc_info=True)

    if conn_to_close is not None:
        try:
            conn_to_close.close()
        except Exception:
            pass

    return state


def main():
    parser = argparse.ArgumentParser(description="Run EvidenceFlow LangGraph workflow with PostgreSQL Checkpointing, Neo4j Memory, & Scoped Cache.")
    parser.add_argument("--query", type=str, required=True, help="Technical question to research")
    parser.add_argument("--version", type=str, default=None, help="Explicit PostgreSQL version (14, 15, 16)")
    parser.add_argument("--thread-id", type=str, default="default_thread", help="Conversation thread ID")
    parser.add_argument("--use-checkpoint", action="store_true", help="Enable persistent PostgreSQL checkpointer")
    parser.add_argument("--enable-memory", action="store_true", help="Enable Neo4j conversational graph memory resolver")
    parser.add_argument("--disable-cache", action="store_true", help="Bypass 5-layer cache gatekeeper")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"\n=======================================================")
    print(f" EvidenceFlow AI - Checkpointed Workflow Execution")
    print(f"=======================================================")
    print(f"Thread ID:      {args.thread_id}")
    print(f"Query:          {args.query}")
    print(f"Version:        {args.version or '(Auto-detect by Planner)'}")
    print(f"Memory Enabled: {args.enable_memory}")
    print(f"Cache Enabled:  {not args.disable_cache}\n")

    state = run_pipeline(
        query=args.query,
        version=args.version,
        thread_id=args.thread_id,
        use_postgres_checkpointer=args.use_checkpoint,
        enable_memory=args.enable_memory,
        enable_cache=not args.disable_cache,
    )

    print(f"[0] Cache Status: {'HIT (' + str(state.cached_layer) + ')' if state.cache_hit else 'MISS'}")
    print(f"[1] Target Scope: PostgreSQL {state.target_version or 'Auto'}")
    if state.plan:
        print(f"[2] Planned Sub-queries: {state.plan}")
    if state.retrieved_passages:
        print(f"\n[3] Retrieved Passages ({len(state.retrieved_passages)}):")
        for i, p in enumerate(state.retrieved_passages, 1):
            score_val = p.get("rrf_score") or p.get("score", 0.0)
            print(f"    ({i}) [{p['chunk_id']}] (Score: {score_val:.4f})")
            print(f"        Section: {p.get('section')}")

    print(f"\n[3.5] Contradiction Analysis:")
    print(f"      Detected: {state.contradiction_detected}")
    if state.contradiction_details:
        print(f"      Relationship: {state.contradiction_details.get('relationship_type')}")
        print(f"      Reason: {state.contradiction_details.get('reason')}")

    print(f"\n[4] Output Answer:\n{state.draft_answer}\n")
    print(f"[5] L1 Verification Result: {'PASSED' if state.l1_passed else 'FAILED'}")
    print(f"    Is Abstained: {state.is_abstained}")
    if state.abstention_reason:
        print(f"    Abstention Reason: {state.abstention_reason}")

    if state.l2_invoked:
        print(f"\n[5.5] L2 Judge Arbitration:")
        print(f"      Invoked:   {state.l2_invoked}")
        print(f"      Passed:    {state.l2_passed}")
        print(f"      Verdict:   {state.l2_verdict}")
        print(f"      Score:     {state.l2_score}")
        print(f"      Reasoning: {state.l2_reasoning}")

    if state.citations:
        print(f"\n[6] Citations ({len(state.citations)}):")
        for c in state.citations:
            if isinstance(c, Citation):
                verified_mark = "[VERIFIED]" if c.is_verified else "[UNVERIFIED]"
                print(f"    - {verified_mark} {c.chunk_id}")
            else:
                verified_mark = "[VERIFIED]" if c.get("is_verified") else "[UNVERIFIED]"
                print(f"    - {verified_mark} {c.get('chunk_id')}")

    print(f"\n[7] Audit Trail Steps ({len(state.audit_trail)}):")
    for step in state.audit_trail:
        print(f"    - Node: {step.get('node'):<22} Action: {step.get('action')}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
