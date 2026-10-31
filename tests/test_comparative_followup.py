"""
test_comparative_followup.py: Synthetic unit tests for Item 5:
  - For comparative follow-ups ("same in 16?", "did it change?"), fetch the chunk of each version
    (reuse the diff_sections logic / multi-version retrieval).
  - Answer with the supported value and say what cannot be confirmed.
  - Do not abstain when the new version's chunk contains the answer.
"""

import uuid
import pytest
from evidenceflow.state import EvidenceFlowState
from evidenceflow.memory.init_neo4j import init_neo4j_schema, drop_graph
from evidenceflow.memory.graph_client import record_turn, resolve_entity_for_turn
from evidenceflow.graph.nodes.retriever import retriever_node
from evidenceflow.graph.nodes.answer_writer import deterministic_fallback_answer, build_grounded_prompt


def test_comparative_retrieval_fetches_both_versions():
    """Verify retriever fetches chunks for both target version and prior version on comparative queries."""
    # Synthetic state: comparing work_mem between 15 and 16
    state = EvidenceFlowState(
        query="Is work_mem's default the same in 16?",
        target_version="16",
        prior_version="15",
        plan=["work_mem default value in PostgreSQL"],
    )

    result = retriever_node(state)
    assert result.get("is_abstained") is False
    passages = result.get("retrieved_passages", [])
    assert len(passages) >= 2

    # Verify both version 16 and version 15 chunks are present in retrieved passages
    chunk_ids = [p["chunk_id"] for p in passages]
    has_v16 = any("16" in cid for cid in chunk_ids)
    has_v15 = any("15" in cid for cid in chunk_ids)
    assert has_v16 is True, f"Expected v16 passage in {chunk_ids}"
    assert has_v15 is True, f"Expected v15 passage in {chunk_ids}"


def test_comparative_offline_answer_does_not_abstain():
    """Verify offline answer writer does not abstain when target version chunk has the answer."""
    passages = [
        {
            "chunk_id": "pg16-runtime-config-resource-work-mem",
            "doc_name": "runtime-config-resource.html",
            "section": "Resource Consumption > Memory",
            "title": "work_mem (integer)",
            "content": "Section: Resource Consumption\nParameter: work_mem (integer)\nSpecifies the baseline amount of memory. The default is 4MB.",
        },
        {
            "chunk_id": "pg15-runtime-config-resource-work-mem",
            "doc_name": "runtime-config-resource.html",
            "section": "Resource Consumption > Memory",
            "title": "work_mem (integer)",
            "content": "Section: Resource Consumption\nParameter: work_mem (integer)\nSpecifies the baseline amount of memory. The default is 4MB.",
        },
    ]

    answer = deterministic_fallback_answer("Is work_mem's default the same in 16?", "16", passages)
    text, code = answer
    assert code is None, f"Expected no abstention, got {code}"
    assert "4MB" in text
    assert "[^pg16-runtime-config-resource-work-mem]" in text
    # Verify formatted across multiple lines to avoid header-only detection
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    assert len(lines) >= 2


def test_build_grounded_prompt_includes_comparative_guideline():
    """Verify prompt guidelines instruct the model not to abstain on comparative queries."""
    prompt = build_grounded_prompt("Is work_mem the same in 16?", "16", [])
    assert "comparative questions across versions" in prompt.lower()
    assert "do not abstain when the target version" in prompt.lower()
