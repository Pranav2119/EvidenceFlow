"""
test_l2_judge.py: Unit and integration tests for Independent L2 Judge node.
Tests selective invocation triggers, arbitration verdicts (AFFIRMED/REJECTED),
deterministic value verification fallback, and the Break It on Purpose test
comparing Selective Invocation vs Global LLM-as-a-Judge invocation.
"""

import pytest
from unittest.mock import MagicMock, patch

from evidenceflow.state import EvidenceFlowState, Citation
from evidenceflow.graph.nodes.judge import (
    should_invoke_l2_judge,
    evaluate_l2_deterministic,
    l2_judge_node,
    build_judge_prompt,
)
from evidenceflow.graph.workflow import create_evidenceflow_graph


def test_selective_invocation_triggers():
    """
    Verifies that L2 Judge is ONLY invoked when specific triggers are met,
    and bypassed for standard, clean single-version queries (ADR-005).
    """
    # 1. Standard clean single-version query -> Bypassed (False)
    standard_state = EvidenceFlowState(
        query="What is work_mem in PostgreSQL 16?",
        target_version="16",
        retrieved_passages=[
            {"chunk_id": "pg16-work-mem", "content": "work_mem defaults to 4MB.", "score": 0.85}
        ],
        citations=[Citation(chunk_id="pg16-work-mem", doc_name="pg16", section="Resource", quote="work_mem", is_verified=True)],
        contradiction_detected=False,
    )
    assert should_invoke_l2_judge(standard_state) is False

    # 2. Trigger: Contradiction Analyst detected cross-version or intra-version contradiction
    contradiction_state = standard_state.model_copy(update={"contradiction_detected": True})
    assert should_invoke_l2_judge(contradiction_state) is True

    # 3. Trigger: Cross-version comparative query (target != prior)
    multiversion_state = standard_state.model_copy(update={"prior_version": "14"})
    assert should_invoke_l2_judge(multiversion_state) is True

    # 4. Trigger: Low retrieval confidence (top passage score < 0.35)
    low_score_state = EvidenceFlowState(
        query="Obscure internal parameter",
        target_version="16",
        retrieved_passages=[
            {"chunk_id": "pg16-obscure", "content": "some text", "score": 0.22}
        ],
        citations=[Citation(chunk_id="pg16-obscure", doc_name="pg16", section="Sec", quote="text", is_verified=True)],
    )
    assert should_invoke_l2_judge(low_score_state) is True

    # 5. Trigger: Complex multi-source claim with 3+ citations
    multi_citation_state = standard_state.model_copy(
        update={
            "citations": [
                Citation(chunk_id="c1", doc_name="d1", section="s1", quote="q1", is_verified=True),
                Citation(chunk_id="c2", doc_name="d2", section="s2", quote="q2", is_verified=True),
                Citation(chunk_id="c3", doc_name="d3", section="s3", quote="q3", is_verified=True),
            ]
        }
    )
    assert should_invoke_l2_judge(multi_citation_state) is True


def test_l2_judge_deterministic_affirmed():
    """
    Verifies that the deterministic evaluation passes and affirms answers
    whose claimed values faithfully match the cited passage text.
    """
    query = "What is the default vacuum_cost_limit in PostgreSQL 16?"
    passages = [
        {
            "chunk_id": "pg16-vacuum-cost-limit",
            "content": "vacuum_cost_limit (integer) The accumulated cost that will cause the vacuuming process to sleep. The default value is 200.",
        }
    ]
    citations = [{"chunk_id": "pg16-vacuum-cost-limit"}]
    draft_answer = "In PostgreSQL 16, vacuum_cost_limit defaults to 200 [^pg16-vacuum-cost-limit]."

    result = evaluate_l2_deterministic(
        query=query,
        draft_answer=draft_answer,
        passages=passages,
        citations=citations,
    )

    assert result["verdict"] == "AFFIRMED"
    assert result["score"] >= 0.9
    assert "faithfully supported" in result["reasoning"]


def test_l2_judge_deterministic_rejected_on_hallucination():
    """
    Verifies that the deterministic evaluation catches hallucinated numbers
    or parameter settings that do not exist in the source evidence.
    """
    query = "What is the default vacuum_cost_limit in PostgreSQL 16?"
    passages = [
        {
            "chunk_id": "pg16-vacuum-cost-limit",
            "content": "vacuum_cost_limit (integer) The default value is 200.",
        }
    ]
    citations = [{"chunk_id": "pg16-vacuum-cost-limit"}]
    # Hallucinated value: 9999 instead of 200
    draft_answer = "In PostgreSQL 16, vacuum_cost_limit defaults to 9999 [^pg16-vacuum-cost-limit]."

    result = evaluate_l2_deterministic(
        query=query,
        draft_answer=draft_answer,
        passages=passages,
        citations=citations,
    )

    assert result["verdict"] == "REJECTED"
    assert result["score"] < 0.5
    assert "unsupported parameter values" in result["reasoning"]
    assert "9999" in result["reasoning"]


@pytest.mark.parametrize(
    "claim, evidence, verdict",
    [("8MB", "128MB", "REJECTED"), ("3", "three", "AFFIRMED"), ("60s", "1min", "AFFIRMED")],
)
def test_deterministic_judge_uses_complete_equivalent_values(claim, evidence, verdict):
    result = evaluate_l2_deterministic(
        query="What is synth_limit?",
        draft_answer=f"synth_limit defaults to {claim} [^chunk1].",
        passages=[{"chunk_id": "chunk1", "content": f"synth_limit defaults to {evidence}."}],
        citations=[{"chunk_id": "chunk1"}],
    )
    assert result["verdict"] == verdict


def test_l2_judge_node_bypassed_for_standard_query():
    """
    Verifies that when should_invoke_l2_judge returns False, the node
    records an l2_bypassed audit action, sets l2_invoked = False, and makes 0 LLM calls.
    """
    state = EvidenceFlowState(
        query="What is work_mem in PostgreSQL 16?",
        target_version="16",
        retrieved_passages=[
            {"chunk_id": "pg16-work-mem", "content": "work_mem defaults to 4MB.", "score": 0.90}
        ],
        draft_answer="work_mem defaults to 4MB [^pg16-work-mem].",
        citations=[Citation(chunk_id="pg16-work-mem", doc_name="pg16", section="Resource", quote="4MB", is_verified=True)],
        l1_passed=True,
        contradiction_detected=False,
    )

    with patch("evidenceflow.graph.nodes.judge.get_gemini_judge_client") as mock_client:
        updates = l2_judge_node(state)
        mock_client.assert_not_called()

    assert updates["l2_invoked"] is False
    assert updates["l2_passed"] is True
    assert updates["audit_trail"][-1]["action"] == "l2_bypassed"


def test_l2_judge_node_llm_affirmation():
    """
    Verifies that when invoked with an active trigger, the node prompts the LLM,
    parses JSON arbitration verdict, and updates state accordingly.
    """
    state = EvidenceFlowState(
        query="Compare vacuum_cost_page_dirty in Postgres 14 vs 16",
        target_version="16",
        prior_version="14",
        contradiction_detected=True,
        contradiction_details={
            "contradiction_detected": True,
            "relationship_type": "SUPERSEDES",
            "reason": "Default changed from 20 to 0 in PG16",
        },
        retrieved_passages=[
            {"chunk_id": "pg14-dirty", "content": "vacuum_cost_page_dirty is 20", "score": 0.8},
            {"chunk_id": "pg16-dirty", "content": "vacuum_cost_page_dirty is 0", "score": 0.8},
        ],
        draft_answer="In PG14, vacuum_cost_page_dirty is 20 [^pg14-dirty]. In PG16, it changed to 0 [^pg16-dirty].",
        citations=[
            Citation(chunk_id="pg14-dirty", doc_name="pg14", section="Resource", quote="20", is_verified=True),
            Citation(chunk_id="pg16-dirty", doc_name="pg16", section="Resource", quote="0", is_verified=True),
        ],
        l1_passed=True,
    )

    mock_llm_response = MagicMock()
    mock_llm_response.text = '```json\n{"verdict": "AFFIRMED", "score": 0.98, "reasoning": "Accurately explains parameter change from 20 to 0 with dual citations."}\n```'

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_llm_response

    with patch("evidenceflow.graph.nodes.judge.get_gemini_judge_client", return_value=mock_client):
        updates = l2_judge_node(state)

    assert updates["l2_invoked"] is True
    assert updates["l2_passed"] is True
    assert updates["l2_verdict"] == "AFFIRMED"
    assert updates["l2_score"] == 0.98
    assert "20 to 0" in updates["l2_reasoning"]
    assert updates["audit_trail"][-1]["action"] == "l2_arbitrated"


def test_l2_judge_node_rejection_triggers_abstention():
    """
    Verifies that when L2 Judge issues a REJECTED verdict, the state is marked as abstained
    with categorical code L2_REJECTED.
    """
    state = EvidenceFlowState(
        query="Compare wal_compression between 14 and 16",
        target_version="16",
        prior_version="14",
        contradiction_detected=True,
        retrieved_passages=[
            {"chunk_id": "pg14-wal", "content": "wal_compression is on or off", "score": 0.8},
            {"chunk_id": "pg16-wal", "content": "wal_compression supports pglz, lz4, zstd", "score": 0.8},
        ],
        draft_answer="wal_compression is identical in both versions.",
        citations=[Citation(chunk_id="pg14-wal", doc_name="pg14", section="WAL", quote="off", is_verified=True)],
        l1_passed=True,
    )

    mock_llm_response = MagicMock()
    mock_llm_response.text = '{"verdict": "REJECTED", "score": 0.1, "reasoning": "Answer erroneously claims wal_compression is identical when PG16 added enum compression methods."}'

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_llm_response

    with patch("evidenceflow.graph.nodes.judge.get_gemini_judge_client", return_value=mock_client):
        updates = l2_judge_node(state)

    assert updates["l2_invoked"] is True
    assert updates["l2_passed"] is False
    assert updates["l2_verdict"] == "REJECTED"
    assert updates["is_abstained"] is True
    assert updates["abstention_code"] == "L2_REJECTED"
    assert "Independent L2 Judge Rejected" in updates["abstention_reason"]


def test_break_it_on_purpose_global_vs_selective_judge():
    """
    Break It on Purpose (Day 15 Section 6):
    Compares Selective Invocation (EvidenceFlow AI) vs Global Invocation (Naive LLM-as-a-Judge)
    across a batch of 20 standard queries.
    
    Proves mathematically that:
      - Selective Invocation: Invoked on 0/20 standard queries (0 API calls, 0ms overhead).
      - Global Invocation: Invoked on 20/20 standard queries (100% API calls, tripling latency/cost).
    """
    standard_queries = [
        f"What is configuration parameter setting_{i} in PostgreSQL 16?"
        for i in range(20)
    ]

    selective_invocations = 0
    global_invocations = 0

    for q in standard_queries:
        state = EvidenceFlowState(
            query=q,
            target_version="16",
            retrieved_passages=[{"chunk_id": f"chunk_{q}", "content": "defaults to 100", "score": 0.88}],
            citations=[Citation(chunk_id=f"chunk_{q}", doc_name="doc", section="sec", quote="100", is_verified=True)],
            l1_passed=True,
            contradiction_detected=False,
        )

        # 1. Selective Invocation evaluation
        if should_invoke_l2_judge(state, force_l2=False):
            selective_invocations += 1

        # 2. Naive Global Invocation evaluation
        if should_invoke_l2_judge(state, force_l2=True):
            global_invocations += 1

    # Invariants under test:
    assert selective_invocations == 0, f"Expected 0 selective invocations, got {selective_invocations}"
    assert global_invocations == 20, f"Expected 20 global invocations, got {global_invocations}"

    # Verify that selective invocation saves 100% of LLM judge calls on standard queries
    call_savings_pct = ((global_invocations - selective_invocations) / global_invocations) * 100
    assert call_savings_pct == 100.0


def test_end_to_end_workflow_with_l2_judge_invocation():
    """
    Tests that the compiled LangGraph workflow selectively executes the L2 Judge
    on a comparative cross-version query and logs the audit trail action.
    """
    initial_state = EvidenceFlowState(
        query="Compare vacuum_cost_page_dirty in Postgres 14 vs 16",
        thread_id="test_judge_thread",
    )

    mock_passages = [
        {"chunk_id": "pg14-dirty", "content": "Parameter: vacuum_cost_page_dirty (integer)\nThe default value is 20.", "score": 0.8, "version": "14", "section": "Resource"},
        {"chunk_id": "pg16-dirty", "content": "Parameter: vacuum_cost_page_dirty (integer)\nThe default value is 0.", "score": 0.8, "version": "16", "section": "Resource"},
    ]

    def mock_retriever(state: EvidenceFlowState, **kwargs):
        return {
            "retrieved_passages": mock_passages,
            "is_abstained": False,
            "audit_trail": list(state.audit_trail) + [{"node": "retriever", "action": "passages_retrieved"}],
        }

    with patch("evidenceflow.graph.workflow.retriever_node", side_effect=mock_retriever):
        app = create_evidenceflow_graph(enable_audit=True, enable_memory=False, enable_cache=False)
        final_state = app.invoke(initial_state)

    # Invariants:
    # 1. Contradiction analyst ran
    assert final_state["target_version"] == "16"
    assert final_state["prior_version"] == "14"
    assert final_state["contradiction_detected"] is True

    # 2. Judge node was selectively invoked
    assert final_state["l2_invoked"] is True
    assert final_state["l2_passed"] is True
    assert final_state["l2_verdict"] in {"AFFIRMED", "REVISED"}

    # 3. Audit trail records l2_arbitrated
    actions = [s.get("action") for s in final_state["audit_trail"]]
    assert "l2_arbitrated" in actions
