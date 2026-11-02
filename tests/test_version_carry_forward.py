"""
test_version_carry_forward.py: Synthetic unit tests for Item 3:
  - Extract any major version number mentioned after version/PostgreSQL/Postgres/PG/v,
    then validate it against the supported set.
  - The resolver must carry 12 forward, not fall back to the prior turn's version.
  - Downstream guardrails must reject unsupported versions with UNSUPPORTED_VERSION.
"""

import uuid
import pytest
from evidenceflow.memory.init_neo4j import init_neo4j_schema, drop_graph
from evidenceflow.memory.graph_client import (
    record_turn,
    resolve_entity_for_turn,
    extract_explicit_version,
    safe_substitute_pronoun,
    SUPPORTED_VERSIONS,
)
from evidenceflow.state import EvidenceFlowState
from evidenceflow.guardrails.input_guardrail import input_guardrail_node


@pytest.mark.parametrize(
    "query, expected_version",
    [
        ("What was its default in version 12?", "12"),
        ("What is work_mem in PostgreSQL 11?", "11"),
        ("What is shared_buffers in PG 10?", "10"),
        ("How about in Postgres 13?", "13"),
        ("What about v12?", "12"),
        ("What is maintenance_work_mem in PostgreSQL 16?", "16"),
        ("And in 15?", "15"),
        ("In 14?", "14"),
        ("In 2024, who won the match?", None),
        ("What is the capital of France?", None),
    ],
)
def test_extract_explicit_version_synthetic(query, expected_version):
    """Verify major version numbers are extracted correctly across supported and unsupported targets."""
    actual = extract_explicit_version(query)
    assert actual == expected_version


def test_safe_substitute_pronoun_no_version_duplication():
    """Verify safe_substitute_pronoun does not create duplicate version suffixes."""
    query = "What was its default in version 12?"
    entity = "maintenance_work_mem"
    rewritten = safe_substitute_pronoun(query, entity, target_version="12")
    assert rewritten is not None
    assert entity in rewritten
    assert "version 12" in rewritten
    # Must not duplicate like 'in version 12 in PostgreSQL 12'
    assert "in PostgreSQL 12" not in rewritten


def test_resolver_carries_unsupported_version_forward():
    """Verify resolver carries forward version 12 instead of falling back to prior turn's version 16."""
    init_neo4j_schema()
    thread_id = f"test-thread-carry-ver-{uuid.uuid4().hex[:8]}"
    turn_1_id = f"turn1_{uuid.uuid4().hex[:8]}"

    try:
        # Seed Turn 1 with prior version "16" and entity "work_mem"
        record_turn(
            turn_id=turn_1_id,
            thread_id=thread_id,
            turn_number=1,
            question="What is work_mem in PostgreSQL 16?",
            version_scope="16",
            entities=[{"name": "work_mem", "type": "SETTING"}],
            claims=[{"id": f"{turn_1_id}_c1", "text": "work_mem defaults to 4MB in PostgreSQL 16.", "verdict": "VERIFIED"}],
            is_abstained=False,
        )

        # Turn 2 asks about version 12
        turn_2_query = "What was its default in version 12?"
        resolution = resolve_entity_for_turn(thread_id, turn_2_query)

        assert resolution["status"] == "resolved"
        assert resolution["entity_name"] == "work_mem"
        # Must carry 12 forward, NOT fall back to 16
        assert resolution["target_version"] == "12"
        assert resolution.get("is_supported_version") is False
        assert "12" in resolution["rewritten_query"]
        assert "16" not in resolution["rewritten_query"]

        # Downstream guardrail must flag unsupported version
        state = EvidenceFlowState(
            query=resolution["rewritten_query"],
            thread_id=thread_id,
            target_version=resolution["target_version"],
        )
        guard_res = input_guardrail_node(state)
        assert guard_res["is_abstained"] is True
        assert guard_res["abstention_code"] == "UNSUPPORTED_VERSION"

    finally:
        drop_graph(thread_id=thread_id)
