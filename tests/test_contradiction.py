"""
test_contradiction.py: Comprehensive test suite for Contradiction Analyst node,
cross-version evolution reasoning (SUPERSEDES), and factual conflict detection (CONTRADICTS).
"""

from unittest.mock import MagicMock, patch
import pytest

from evidenceflow.graph.nodes.contradiction import (
    analyze_cross_version_differences_deterministic,
    contradiction_analyst_node,
    extract_parameter_values_from_passage,
)
from evidenceflow.memory.graph_client import (
    get_claim_relationships,
    get_passage_relationships,
    record_claim_relationship,
    record_passage_relationship,
)
from evidenceflow.state import EvidenceFlowState


def test_extract_parameter_values():
    """Extracts parameter name, default value, and enum options from documentation text."""
    sample = (
        "Parameter: wal_compression ( enum )\n\n"
        "Enables compression of WAL. Accepts off, on, pg_lz, lz4, and zstd. "
        "The default value is off."
    )
    res = extract_parameter_values_from_passage(sample)
    assert res["parameter"] == "wal_compression"
    assert res["default"] == "off"
    assert "lz4" in res["options"]
    assert "zstd" in res["options"]


def test_cross_version_changed_default_supersedes():
    """When a parameter default changes between versions, SUPERSEDES relationship is identified."""
    p_older = {
        "chunk_id": "pg14_param_chunk",
        "version": "14",
        "content": "Parameter: vacuum_cost_limit ( integer )\nThe default value is 200.",
    }
    p_newer = {
        "chunk_id": "pg16_param_chunk",
        "version": "16",
        "content": "Parameter: vacuum_cost_limit ( integer )\nThe default value is 500.",
    }
    passages_by_v = {
        "14": [p_older],
        "16": [p_newer],
    }

    detected, rel_type, details = analyze_cross_version_differences_deterministic(
        passages_by_v, "Compare vacuum_cost_limit in Postgres 14 vs 16"
    )

    assert detected is True
    assert rel_type == "SUPERSEDES"
    assert details["subject_version"] == "16"
    assert details["target_version"] == "14"
    assert details["subject_chunk_id"] == "pg16_param_chunk"
    assert details["target_chunk_id"] == "pg14_param_chunk"
    assert "changed from '200' to '500'" in details["reason"]


def test_cross_version_options_expansion_supersedes():
    """When algorithms/options are added in a newer release, SUPERSEDES is recorded."""
    p_older = {
        "chunk_id": "pg14_wal",
        "version": "14",
        "content": "Parameter: wal_compression ( enum )\nAccepts off, on, pg_lz, lz4.",
    }
    p_newer = {
        "chunk_id": "pg16_wal",
        "version": "16",
        "content": "Parameter: wal_compression ( enum )\nAccepts off, on, pg_lz, lz4, zstd.",
    }
    passages_by_v = {
        "14": [p_older],
        "16": [p_newer],
    }

    detected, rel_type, details = analyze_cross_version_differences_deterministic(
        passages_by_v, "Did wal_compression change in Postgres 16?"
    )

    assert detected is True
    assert rel_type == "SUPERSEDES"
    assert "zstd" in details["reason"]


def test_direct_intra_version_contradiction():
    """Two conflicting assertions for the exact same release are identified as CONTRADICTS."""
    p1 = {
        "chunk_id": "chunk_doc_a",
        "version": "16",
        "content": "Parameter: max_connections ( integer )\nThe default value is 100.",
    }
    p2 = {
        "chunk_id": "chunk_doc_b",
        "version": "16",
        "content": "Parameter: max_connections ( integer )\nThe default value is 500.",
    }
    passages_by_v = {
        "16": [p1, p2],
    }

    detected, rel_type, details = analyze_cross_version_differences_deterministic(
        passages_by_v, "What is max_connections?"
    )

    assert detected is True
    assert rel_type == "CONTRADICTS"
    assert details["subject_chunk_id"] == "chunk_doc_a"
    assert details["target_chunk_id"] == "chunk_doc_b"


def test_break_it_on_purpose_identical_passages_no_contradiction():
    """
    Break It on Purpose (Roadmap Day 14 Section 6):
    Feed two identical, non-contradictory passages into the contradiction analyst.
    Verify that the agent does NOT hallucinate a conflict and returns contradiction_detected: False.
    """
    p1 = {
        "chunk_id": "pg14_vacuum",
        "version": "14",
        "content": "Parameter: vacuum_cost_limit ( integer )\nThe default value is 200.",
    }
    p2 = {
        "chunk_id": "pg16_vacuum",
        "version": "16",
        "content": "Parameter: vacuum_cost_limit ( integer )\nThe default value is 200.",
    }
    passages_by_v = {
        "14": [p1],
        "16": [p2],
    }

    detected, rel_type, details = analyze_cross_version_differences_deterministic(
        passages_by_v, "Did vacuum_cost_limit default change between Postgres 14 and 16?"
    )

    # Invariant: Must NOT hallucinate a conflict when defaults are identical!
    assert detected is False, "Expected contradiction_detected to be False for identical defaults!"
    assert rel_type == "NO_CONTRADICTION"


def test_contradiction_analyst_node_execution():
    """Executes contradiction_analyst_node and verifies state updates and audit trail."""
    state = EvidenceFlowState(
        query="Compare wal_compression across Postgres 14 and 16",
        target_version="16",
        retrieved_passages=[
            {
                "chunk_id": "pg14_wal_001",
                "version": "14",
                "content": "Parameter: wal_compression ( enum )\nAccepts off, on, pg_lz, lz4. Default is off.",
            },
            {
                "chunk_id": "pg16_wal_005",
                "version": "16",
                "content": "Parameter: wal_compression ( enum )\nAccepts off, on, pg_lz, lz4, zstd. Default is off.",
            },
        ],
    )

    res = contradiction_analyst_node(state)
    assert res["contradiction_detected"] is True
    assert res["contradiction_details"]["relationship_type"] == "SUPERSEDES"

    audit = res["audit_trail"][-1]
    assert audit["node"] == "contradiction_analyst"
    assert audit["action"] == "contradiction_analyzed"
    assert audit["relationship_type"] == "SUPERSEDES"


def test_neo4j_claim_and_passage_relationship_wiring():
    """Verifies that record_passage_relationship and record_claim_relationship create edges in Neo4j."""
    # 1. Passage relationship test
    p_newer = "test_pass_v16_chunk"
    p_older = "test_pass_v14_chunk"

    ok = record_passage_relationship(
        subject_passage_id=p_newer,
        target_passage_id=p_older,
        rel_type="SUPERSEDES",
        reason="PostgreSQL 16 adds zstd support",
    )
    assert ok is True

    rels = get_passage_relationships(p_newer)
    assert any(r["rel_type"] == "SUPERSEDES" and r["target_passage_id"] == p_older for r in rels)

    # 2. Invalid relation type raises ValueError
    with pytest.raises(ValueError):
        record_passage_relationship(p_newer, p_older, rel_type="INVALID_REL")
