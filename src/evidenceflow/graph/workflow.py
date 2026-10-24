"""
workflow.py: EvidenceFlow LangGraph State Machine assembly and compilation.
Wires START -> planner -> retriever -> answer_writer -> verifier -> END.
"""

import argparse
import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END
from evidenceflow.state import EvidenceFlowState, Citation
from evidenceflow.graph.nodes import (
    planner_node,
    retriever_node,
    answer_writer_node,
    verifier_node,
)

logger = logging.getLogger(__name__)

# D3-10: Cached compiled graph singleton to avoid re-compiling per pipeline invocation
_compiled_app = None


def create_evidenceflow_graph():
    """
    Constructs and compiles the baseline 4-node sequential LangGraph workflow.
    """
    builder = StateGraph(EvidenceFlowState)

    # 1. Register Nodes
    builder.add_node("planner", planner_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("answer_writer", answer_writer_node)
    builder.add_node("verifier", verifier_node)

    # 2. Register Linear Edges
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "answer_writer")
    builder.add_edge("answer_writer", "verifier")
    builder.add_edge("verifier", END)

    # 3. Compile Graph
    return builder.compile()


def get_compiled_graph():
    """
    Returns the cached compiled StateGraph application instance.
    D3-10: Compiles once instead of re-building on every query.
    """
    global _compiled_app
    if _compiled_app is None:
        _compiled_app = create_evidenceflow_graph()
    return _compiled_app


def run_pipeline(query: str, version: Optional[str] = None) -> EvidenceFlowState:
    """
    Executes a single pass through the compiled LangGraph workflow.
    """
    app = get_compiled_graph()
    initial_input = {"query": query}
    if version:
        initial_input["target_version"] = version

    result = app.invoke(initial_input)
    # Ensure return is Pydantic instance for clean attribute access
    if isinstance(result, dict):
        return EvidenceFlowState(**result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run EvidenceFlow baseline LangGraph workflow.")
    parser.add_argument("--query", type=str, required=True, help="Technical question to research")
    parser.add_argument("--version", type=str, default=None, help="Explicit PostgreSQL version (14, 15, 16)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"\n=======================================================")
    print(f" EvidenceFlow AI - Baseline Workflow Execution")
    print(f"=======================================================")
    print(f"Query:   {args.query}")
    print(f"Version: {args.version or '(Auto-detect by Planner)'}\n")

    state = run_pipeline(query=args.query, version=args.version)

    print(f"[1] Target Scope: PostgreSQL {state.target_version}")
    print(f"[2] Planned Sub-queries: {state.plan}")
    print(f"\n[3] Retrieved Passages ({len(state.retrieved_passages)}):")
    for i, p in enumerate(state.retrieved_passages, 1):
        score_val = p.get("rrf_score") or p.get("score", 0.0)
        print(f"    ({i}) [{p['chunk_id']}] (Score: {score_val:.4f})")
        print(f"        Section: {p.get('section')}")

    print(f"\n[4] Draft Answer:\n{state.draft_answer}\n")
    print(f"[5] L1 Verification Result: {'PASSED' if state.l1_passed else 'FAILED'}")
    print(f"    Is Abstained: {state.is_abstained}")
    if state.abstention_reason:
        print(f"    Abstention Reason: {state.abstention_reason}")

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
        print(f"    - Node: {step.get('node'):<14} Action: {step.get('action')}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
