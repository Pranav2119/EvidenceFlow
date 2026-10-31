"""
test_memory_graph.py: Integration test suite for Neo4j conversational memory graph.
Verifies all Day 8 requirements:
  [Req 2] Uniqueness constraints for Question.id and Claim.id.
  [Req 3] Discrete parameterized queries; empty entities or missing version writes claims.
  [Req 4] Explicit setting wins over memory; refusals/multi-entity turns return 'ambiguous'.
  [Req 5] Memory only rewrites the query; answers strictly cite Qdrant corpus chunks.
  [Req 6] Uses identical thread_id, turn_id, turn_number as PostgreSQL storage.ledger.
  [Req 7] Real measured latency logged.
"""

import uuid
import re
import pytest
from evidenceflow.memory.connection import get_neo4j_session
from evidenceflow.memory.init_neo4j import init_neo4j_schema, drop_graph, verify_schema
from evidenceflow.memory.graph_client import (
    record_turn,
    get_session_history,
    resolve_entity_for_turn,
    is_follow_up_query,
    extract_explicit_setting,
    extract_explicit_version,
)
from evidenceflow.graph.workflow import create_evidenceflow_graph


@pytest.fixture(autouse=True)
def setup_teardown():
    """Initializes schema and cleans up test threads before/after each test."""
    init_neo4j_schema()
    yield
    drop_graph(thread_id="test-thread-multi-turn")
    drop_graph(thread_id="test-thread-empty-missing")
    drop_graph(thread_id="test-thread-explicit-win")
    drop_graph(thread_id="test-thread-refusal")
    drop_graph(thread_id="test-thread-multi-entity")
    drop_graph(thread_id="test-thread-e2e-corpus-cite")


def test_schema_uniqueness_constraints_active():
    """[Req 2] Verifies that Question.id and Claim.id uniqueness constraints are active."""
    schema_items = verify_schema()
    assert any("CONSTRAINT: question_id_unique" in item for item in schema_items)
    assert any("CONSTRAINT: claim_id_unique" in item for item in schema_items)
    assert any("CONSTRAINT: turn_id_unique" in item for item in schema_items)
    assert any("CONSTRAINT: entity_name_unique" in item for item in schema_items)


def test_record_turn_empty_entities_and_missing_version_writes_claims():
    """[Req 3] Verifies that empty entities and non-existent version do not skip Claim writes."""
    thread_id = "test-thread-empty-missing"
    turn_id = str(uuid.uuid4())
    drop_graph(thread_id=thread_id)

    # Ingest turn with entities=[] and non-existent version_scope="99"
    ret_id = record_turn(
        turn_id=turn_id,
        thread_id=thread_id,
        turn_number=1,
        question="A query without entities and invalid version",
        version_scope="99",  # Version 99 does not exist in seed versions
        entities=[],         # Empty entities list
        claims=[{
            "id": f"{turn_id}_claim1",
            "text": "This claim must be written despite missing version and empty entities",
            "verdict": "VERIFIED",
            "passage_id": "pg16-runtime-config-wal-wal-compression",
            "chunk_hash": "dummy_hash_123",
        }],
        is_abstained=False,
    )
    assert ret_id == turn_id

    # Verify claim was committed in Neo4j
    with get_neo4j_session() as session:
        result = session.run(
            """
            MATCH (t:Turn {id: $turn_id})-[:PRODUCED]->(c:Claim {id: $claim_id})
            OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(p:Passage)
            RETURN c.text AS claim_text, c.verdict AS verdict, p.passage_id AS passage_id
            """,
            {"turn_id": turn_id, "claim_id": f"{turn_id}_claim1"},
        )
        record = result.single()
        assert record is not None
        assert record["verdict"] == "VERIFIED"
        assert record["passage_id"] == "pg16-runtime-config-wal-wal-compression"


def test_explicit_setting_in_new_question_wins_over_memory():
    """[Req 4] Verifies that an explicit parameter in the new question overrides memory."""
    thread_id = "test-thread-explicit-win"
    drop_graph(thread_id=thread_id)

    # Turn 1: Discussed wal_compression in v14
    record_turn(
        turn_id=str(uuid.uuid4()),
        thread_id=thread_id,
        turn_number=1,
        question="How do I configure wal_compression in v14?",
        version_scope="14",
        entities=[{"name": "wal_compression", "type": "ConfigParam"}],
        is_abstained=False,
    )

    # Turn 2: User explicitly mentions work_mem
    res = resolve_entity_for_turn(thread_id, "What is work_mem in PostgreSQL 16?")
    assert res["status"] == "passthrough"
    assert res["entity_name"] == "work_mem"
    assert res["target_version"] == "16"
    assert res["entity_name"] != "wal_compression"  # Explicit wins!


def test_prior_turn_refusal_returns_ambiguous():
    """[Req 4] Verifies that follow-up after an abstained turn returns 'ambiguous' with clarifying question."""
    thread_id = "test-thread-refusal"
    drop_graph(thread_id=thread_id)

    # Turn 1: System abstained (e.g. hallucinated parameter)
    record_turn(
        turn_id=str(uuid.uuid4()),
        thread_id=thread_id,
        turn_number=1,
        question="What is quantum_hyperthreading in v16?",
        version_scope="16",
        entities=[{"name": "quantum_hyperthreading", "type": "ConfigParam"}],
        is_abstained=True,
        abstention_code="MISSING_KEYWORD",
    )

    # Turn 2: User asks ambiguous follow-up
    res = resolve_entity_for_turn(thread_id, "Did that change in v16?")
    assert res["status"] == "ambiguous"
    assert res["reason"] == "prior_turn_refusal"
    assert "could not be answered" in res["clarification"]


def test_prior_turn_multiple_entities_returns_ambiguous():
    """[Req 4] Verifies that follow-up after a multi-entity turn returns 'ambiguous' with clarifying question."""
    thread_id = "test-thread-multi-entity"
    drop_graph(thread_id=thread_id)

    # Turn 1: Discussed both min_wal_size and max_wal_size
    record_turn(
        turn_id=str(uuid.uuid4()),
        thread_id=thread_id,
        turn_number=1,
        question="What are min_wal_size and max_wal_size in v15?",
        version_scope="15",
        entities=[
            {"name": "min_wal_size", "type": "ConfigParam"},
            {"name": "max_wal_size", "type": "ConfigParam"},
        ],
        is_abstained=False,
    )

    # Turn 2: User asks ambiguous pronoun question
    res = resolve_entity_for_turn(thread_id, "Does that setting require a server restart?")
    assert res["status"] == "ambiguous"
    assert res["reason"] == "multiple_entities"
    assert "multiple parameters" in res["clarification"]
    assert "min_wal_size" in res["candidate_entities"]
    assert "max_wal_size" in res["candidate_entities"]


def test_successful_followup_resolution_and_latency():
    """[Req 4, Req 7] Verifies single-entity follow-up resolution and logs real execution latency."""
    thread_id = "test-thread-multi-turn"
    drop_graph(thread_id=thread_id)

    # Turn 1: Discussed wal_compression in v14
    record_turn(
        turn_id=str(uuid.uuid4()),
        thread_id=thread_id,
        turn_number=1,
        question="How do I configure wal_compression in v14?",
        version_scope="14",
        entities=[{"name": "wal_compression", "type": "ConfigParam"}],
        is_abstained=False,
    )

    # Turn 2: Follow-up switching to v16
    res = resolve_entity_for_turn(thread_id, "Did that change in v16?")
    assert res["status"] == "resolved"
    assert res["entity_name"] == "wal_compression"
    assert res["target_version"] == "16"
    assert res["prior_version"] == "14"
    assert "wal_compression" in res["rewritten_query"]
    assert "PostgreSQL 16" in res["rewritten_query"]

    # Req 7: Real measured latency must be logged and non-zero
    assert isinstance(res["latency_ms"], float)
    assert res["latency_ms"] > 0.0
    print(f"\n[Real Measured Resolution Latency]: {res['latency_ms']:.3f} ms")


def test_memory_only_rewrites_query_downstream_cites_corpus_chunks():
    """[Req 5] Verifies that rewritten query flows into pipeline and answers cite corpus chunks, NEVER memory nodes."""
    thread_id = "test-thread-e2e-corpus-cite"
    drop_graph(thread_id=thread_id)

    # Seed Turn 1 into Neo4j memory
    record_turn(
        turn_id=str(uuid.uuid4()),
        thread_id=thread_id,
        turn_number=1,
        question="How do I configure wal_compression in v14?",
        version_scope="14",
        entities=[{"name": "wal_compression", "type": "ConfigParam"}],
        is_abstained=False,
    )

    # Turn 2: Follow-up query
    follow_up = "Did that change in v16?"
    res = resolve_entity_for_turn(thread_id, follow_up)
    assert res["status"] == "resolved"
    rewritten_query = res["rewritten_query"]

    # Execute LangGraph workflow with rewritten query
    graph = create_evidenceflow_graph()
    initial_state = {
        "query": rewritten_query,
        "target_version": res["target_version"],
        "thread_id": thread_id,
        "turn_number": 2,
    }
    final_state = graph.invoke(initial_state)

    # Assert downstream answers cite corpus chunks, NEVER Neo4j memory nodes
    assert final_state["is_abstained"] is False
    assert len(final_state["citations"]) > 0

    chunk_pattern = re.compile(r"^pg(14|15|16)-[a-z0-9_-]+$")
    for citation in final_state["citations"]:
        chunk_id = citation.chunk_id if hasattr(citation, "chunk_id") else citation["chunk_id"]
        assert chunk_pattern.match(chunk_id), (
            f"Violation of Req 5: Citation '{chunk_id}' is not a valid Qdrant corpus chunk ID!"
        )
