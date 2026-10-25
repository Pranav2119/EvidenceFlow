"""
workflow.py: EvidenceFlow LangGraph State Machine assembly and compilation.
Wires START -> input_guardrail -> planner -> retriever -> answer_writer -> verifier -> END
with conditional branching to abstention_node for security, domain, and grounding failures.
Integrates PostgreSQL checkpointer for process-restart survival and audit wrapper for immutable ledgers.
"""

import argparse
import logging
import uuid
from typing import Any, Dict, Optional
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START, END

from evidenceflow.state import EvidenceFlowState, Citation
from evidenceflow.guardrails.input_guardrail import input_guardrail_node
from evidenceflow.graph.nodes import (
    planner_node,
    retriever_node,
    answer_writer_node,
    verifier_node,
    abstention_node,
)
from evidenceflow.graph.audit_wrapper import audit_node_wrapper
from evidenceflow.storage.ledger import (
    get_postgres_connection_string,
    record_turn,
)

logger = logging.getLogger(__name__)

# Cached compiled graph singleton (in-memory default)
_compiled_app = None


# --- Conditional Routing Functions ---

def route_after_input_guardrail(state: EvidenceFlowState) -> str:
    """Routes to abstention if input guardrail detected injection or off-domain query."""
    if state.is_abstained:
        return "abstention"
    return "planner"


def route_after_retriever(state: EvidenceFlowState) -> str:
    """Routes to abstention if retriever found no evidence or failed score/keyword threshold."""
    if state.is_abstained:
        return "abstention"
    return "answer_writer"


def route_after_verifier(state: EvidenceFlowState) -> str:
    """Routes to abstention if L1 citation check or L1.5 value presence check failed."""
    if not state.l1_passed or state.is_abstained:
        return "abstention"
    return END


def create_evidenceflow_graph(checkpointer: Optional[Any] = None, enable_audit: bool = True):
    """
    Constructs and compiles the 6-node LangGraph workflow with conditional abstention routes,
    optional transparent audit ledger wrappers, and optional persistent checkpointer.
    """
    builder = StateGraph(EvidenceFlowState)

    # 1. Register Nodes (transparently wrapped with audit ledger logging)
    if enable_audit:
        builder.add_node("input_guardrail", audit_node_wrapper(input_guardrail_node, "input_guardrail"))
        builder.add_node("planner", audit_node_wrapper(planner_node, "planner"))
        builder.add_node("retriever", audit_node_wrapper(retriever_node, "retriever"))
        builder.add_node("answer_writer", audit_node_wrapper(answer_writer_node, "answer_writer"))
        builder.add_node("verifier", audit_node_wrapper(verifier_node, "verifier"))
        builder.add_node("abstention", audit_node_wrapper(abstention_node, "abstention"))
    else:
        builder.add_node("input_guardrail", input_guardrail_node)
        builder.add_node("planner", planner_node)
        builder.add_node("retriever", retriever_node)
        builder.add_node("answer_writer", answer_writer_node)
        builder.add_node("verifier", verifier_node)
        builder.add_node("abstention", abstention_node)

    # 2. Register Edges with Conditional Branches
    builder.add_edge(START, "input_guardrail")
    builder.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {"abstention": "abstention", "planner": "planner"},
    )
    builder.add_edge("planner", "retriever")
    builder.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {"abstention": "abstention", "answer_writer": "answer_writer"},
    )
    builder.add_edge("answer_writer", "verifier")
    builder.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"abstention": "abstention", END: END},
    )
    builder.add_edge("abstention", END)

    # 3. Compile Graph with Checkpointer
    return builder.compile(checkpointer=checkpointer)


def get_compiled_graph():
    """
    Returns the cached compiled StateGraph application instance (in-memory).
    """
    global _compiled_app
    if _compiled_app is None:
        _compiled_app = create_evidenceflow_graph(enable_audit=True)
    return _compiled_app


def run_pipeline(
    query: str,
    version: Optional[str] = None,
    thread_id: str = "default_thread",
    checkpointer: Optional[Any] = None,
    use_postgres_checkpointer: bool = False,
    turn_id: Optional[str] = None,
    turn_number: int = 1,
) -> EvidenceFlowState:
    """
    Executes a single pass through the LangGraph workflow, persisting thread checkpoints
    and audit ledger rows into PostgreSQL.
    """
    turn_uuid = turn_id or str(uuid.uuid4())
    conn_to_close = None

    if checkpointer is None and use_postgres_checkpointer:
        conn_str = get_postgres_connection_string()
        conn_to_close = psycopg.connect(conn_str, autocommit=True)
        checkpointer = PostgresSaver(conn_to_close)

    if checkpointer is not None:
        app = create_evidenceflow_graph(checkpointer=checkpointer, enable_audit=True)
    else:
        app = get_compiled_graph()

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

    # Record turn record in PostgreSQL turns table
    try:
        record_turn(
            turn_id=turn_uuid,
            thread_id=thread_id,
            turn_number=turn_number,
            target_version=state.target_version or (version or "16"),
            query=query,
            draft_answer=state.draft_answer,
            is_abstained=state.is_abstained,
            abstention_code=state.abstention_code,
            abstention_reason=state.abstention_reason,
        )
    except Exception as e:
        logger.error("Failed to record turn to PostgreSQL turns table: %s", e)

    if conn_to_close is not None:
        try:
            conn_to_close.close()
        except Exception:
            pass

    return state


def main():
    parser = argparse.ArgumentParser(description="Run EvidenceFlow LangGraph workflow with PostgreSQL Checkpointing.")
    parser.add_argument("--query", type=str, required=True, help="Technical question to research")
    parser.add_argument("--version", type=str, default=None, help="Explicit PostgreSQL version (14, 15, 16)")
    parser.add_argument("--thread-id", type=str, default="default_thread", help="Conversation thread ID")
    parser.add_argument("--use-checkpoint", action="store_true", help="Enable persistent PostgreSQL checkpointer")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"\n=======================================================")
    print(f" EvidenceFlow AI - Checkpointed Workflow Execution")
    print(f"=======================================================")
    print(f"Thread ID: {args.thread_id}")
    print(f"Query:     {args.query}")
    print(f"Version:   {args.version or '(Auto-detect by Planner)'}\n")

    state = run_pipeline(
        query=args.query,
        version=args.version,
        thread_id=args.thread_id,
        use_postgres_checkpointer=args.use_checkpoint,
    )

    print(f"[1] Target Scope: PostgreSQL {state.target_version or 'Auto'}")
    if state.plan:
        print(f"[2] Planned Sub-queries: {state.plan}")
    if state.retrieved_passages:
        print(f"\n[3] Retrieved Passages ({len(state.retrieved_passages)}):")
        for i, p in enumerate(state.retrieved_passages, 1):
            score_val = p.get("rrf_score") or p.get("score", 0.0)
            print(f"    ({i}) [{p['chunk_id']}] (Score: {score_val:.4f})")
            print(f"        Section: {p.get('section')}")

    print(f"\n[4] Output Answer:\n{state.draft_answer}\n")
    print(f"[5] L1 Verification Result: {'PASSED' if state.l1_passed else 'FAILED'}")
    print(f"    Is Abstained: {state.is_abstained}")
    if state.abstention_reason:
        print(f"    Abstention Reason: {state.abstention_reason}")

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
        print(f"    - Node: {step.get('node'):<16} Action: {step.get('action')}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
