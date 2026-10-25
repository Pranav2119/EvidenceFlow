"""
test_postgres_ledger.py: Comprehensive test suite for Day 5 PostgreSQL Checkpointer & Ledger.
Tests:
  1. Append-only trigger enforcement (blocking UPDATE, DELETE, TRUNCATE).
  2. Process restart persistence & thread isolation (recreating connection & graph).
  3. Step ledger and claim-evidence audit recording with latency, tokens, and hashes.
  4. Secrets exclusion verification across all audit tables.
"""

import uuid
import pytest
import psycopg
from psycopg.errors import RaiseException
from langgraph.checkpoint.postgres import PostgresSaver

from evidenceflow.graph.workflow import create_evidenceflow_graph, run_pipeline
from evidenceflow.storage.ledger import (
    get_postgres_connection_string,
    get_connection_pool,
    close_connection_pool,
    record_turn,
    record_step,
    record_claim,
    fetch_turns,
    fetch_steps,
    fetch_claims,
)
from evidenceflow.storage.migrate_postgres import run_migrations


@pytest.fixture(scope="module", autouse=True)
def setup_database_schema():
    """Ensures migrations and checkpointer tables are initialized before tests run."""
    run_migrations()
    yield
    close_connection_pool()


# --- 1. Test Append-Only Immutability Triggers ---

def test_append_only_triggers_reject_updates_and_deletes():
    """
    Day 5 Req 1 & 7: Verify database triggers reject UPDATE and DELETE operations
    on turns, step_ledger, and claim_audit.
    """
    test_id = str(uuid.uuid4())
    test_thread = f"append-test-{uuid.uuid4().hex[:6]}"

    # Insert baseline records
    record_turn(
        turn_id=test_id,
        thread_id=test_thread,
        turn_number=1,
        target_version="16",
        query="Test query",
        draft_answer="Test answer",
    )
    step_id = record_step(
        turn_id=test_id,
        thread_id=test_thread,
        step_number=1,
        node="test_node",
        action="test_action",
        status="success",
        decided_by="rule",
        latency_ms=12.5,
    )
    claim_id = record_claim(
        turn_id=test_id,
        thread_id=test_thread,
        statement="Test statement",
        chunk_id="pg16-chunk-test",
        chunk_sha256="abc123sha256hash",
        chunk_version="16",
        verifier_verdict="PASS",
        verifier_code="VERIFIED",
    )

    conn_str = get_postgres_connection_string()
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            # 1. Test UPDATE on step_ledger
            with pytest.raises(RaiseException) as exc_info:
                cur.execute("UPDATE step_ledger SET node = 'tampered' WHERE step_id = %s;", (step_id,))
            assert "append-only" in str(exc_info.value).lower()
            conn.rollback()

            # 2. Test DELETE on step_ledger
            with pytest.raises(RaiseException) as exc_info:
                cur.execute("DELETE FROM step_ledger WHERE step_id = %s;", (step_id,))
            assert "append-only" in str(exc_info.value).lower()
            conn.rollback()

            # 3. Test UPDATE on claim_audit
            with pytest.raises(RaiseException) as exc_info:
                cur.execute("UPDATE claim_audit SET verifier_verdict = 'FAIL' WHERE claim_id = %s;", (claim_id,))
            assert "append-only" in str(exc_info.value).lower()
            conn.rollback()

            # 4. Test DELETE on claim_audit
            with pytest.raises(RaiseException) as exc_info:
                cur.execute("DELETE FROM claim_audit WHERE claim_id = %s;", (claim_id,))
            assert "append-only" in str(exc_info.value).lower()
            conn.rollback()

            # 5. Test UPDATE on turns
            with pytest.raises(RaiseException) as exc_info:
                cur.execute("UPDATE turns SET query = 'altered' WHERE turn_id = %s;", (test_id,))
            assert "append-only" in str(exc_info.value).lower()
            conn.rollback()

            # 6. Test DELETE on turns
            with pytest.raises(RaiseException) as exc_info:
                cur.execute("DELETE FROM turns WHERE turn_id = %s;", (test_id,))
            assert "append-only" in str(exc_info.value).lower()
            conn.rollback()


# --- 2. Test Process-Restart Persistence & Thread Isolation ---

def test_process_restart_persistence_and_thread_isolation():
    """
    Day 5 Req 6: Recreate the graph AND the database connection, then verify
    the same thread_id is recovered, and an isolated second thread cannot see it.
    """
    thread_alpha = f"thread-alpha-{uuid.uuid4().hex[:8]}"
    thread_beta = f"thread-beta-{uuid.uuid4().hex[:8]}"
    conn_str = get_postgres_connection_string()

    # Step 1: Run graph on thread_alpha using conn_1
    conn_1 = psycopg.connect(conn_str, autocommit=True)
    checkpointer_1 = PostgresSaver(conn_1)
    graph_1 = create_evidenceflow_graph(checkpointer=checkpointer_1, enable_audit=True)

    input_alpha = {
        "query": "What is the default value of max_connections in PostgreSQL 16?",
        "target_version": "16",
        "thread_id": thread_alpha,
    }
    config_alpha = {"configurable": {"thread_id": thread_alpha}}
    res_alpha = graph_1.invoke(input_alpha, config=config_alpha)
    assert res_alpha["l1_passed"] is True

    # Step 2: SIMULATE COMPLETE PROCESS RESTART
    # Destroy all objects, close connections and connection pools
    conn_1.close()
    del graph_1
    del checkpointer_1
    del conn_1
    close_connection_pool()

    # Step 3: Reconnect with brand-new connection and checkpointer instance
    conn_2 = psycopg.connect(conn_str, autocommit=True)
    checkpointer_2 = PostgresSaver(conn_2)
    graph_2 = create_evidenceflow_graph(checkpointer=checkpointer_2, enable_audit=True)

    # Verify thread_alpha state was persisted to PostgreSQL and recovered
    recovered_alpha_tuple = checkpointer_2.get(config_alpha)
    assert recovered_alpha_tuple is not None, "Failed to recover thread_alpha checkpoint from PostgreSQL"

    state_alpha = graph_2.get_state(config_alpha)
    assert state_alpha is not None and state_alpha.values, "Failed to recover thread_alpha state from PostgreSQL after simulated restart"
    assert state_alpha.values.get("thread_id") == thread_alpha
    assert state_alpha.values.get("target_version") == "16"
    assert state_alpha.values.get("l1_passed") is True
    assert len(state_alpha.values.get("retrieved_passages", [])) > 0

    # Step 4: Verify thread_beta is completely isolated and sees NO state
    config_beta = {"configurable": {"thread_id": thread_beta}}
    recovered_beta_tuple = checkpointer_2.get(config_beta)
    assert recovered_beta_tuple is None, "Thread isolation failed: thread_beta unexpectedly saw thread_alpha state"
    state_beta_initial = graph_2.get_state(config_beta)
    assert state_beta_initial.values == {}, "Thread isolation failed: thread_beta initial state is non-empty"

    # Step 5: Execute thread_beta on a separate query and verify both co-exist independently
    input_beta = {
        "query": "What is the default of fsync in Postgres 16?",
        "target_version": "16",
        "thread_id": thread_beta,
    }
    res_beta = graph_2.invoke(input_beta, config=config_beta)
    assert res_beta["l1_passed"] is True

    # Confirm both threads now exist with distinct states in PostgreSQL
    final_alpha = graph_2.get_state(config_alpha)
    final_beta = graph_2.get_state(config_beta)
    assert final_alpha is not None and final_beta is not None
    assert final_alpha.values["query"] != final_beta.values["query"]
    assert "max_connections" in final_alpha.values["query"]
    assert "fsync" in final_beta.values["query"]

    conn_2.close()


# --- 3. Test Step Ledger & Claim Audit Details Recording ---

def test_step_ledger_and_claim_audit_records():
    """
    Day 5 Req 4 & 5: Ensure node transitions are logged to step_ledger with latency,
    status, decided_by, and claims are logged to claim_audit with chunk hashes.
    """
    thread_id = f"test-audit-ledger-{uuid.uuid4().hex[:6]}"
    state = run_pipeline(
        query="What is the default of shared_buffers in Postgres 16?",
        version="16",
        thread_id=thread_id,
        use_postgres_checkpointer=True,
    )
    assert not state.is_abstained

    # 1. Verify turns table
    turns = fetch_turns(thread_id)
    assert len(turns) >= 1
    t = turns[0]
    assert t["thread_id"] == thread_id
    assert "shared_buffers" in t["query"]
    assert not t["is_abstained"]
    turn_id = t["turn_id"]

    # 2. Verify step_ledger table
    steps = fetch_steps(thread_id, turn_id=turn_id)
    assert len(steps) >= 4  # input_guardrail, planner, retriever, answer_writer, verifier
    node_names = [s["node"] for s in steps]
    assert "input_guardrail" in node_names
    assert "retriever" in node_names
    assert "answer_writer" in node_names
    assert "verifier" in node_names

    for s in steps:
        assert s["latency_ms"] >= 0.0
        assert s["status"] in ("success", "abstained")
        assert s["decided_by"] in ("rule", "llm", "fallback")
        assert isinstance(s["details"], dict)

    # 3. Verify claim_audit table
    claims = fetch_claims(thread_id, turn_id=turn_id)
    assert len(claims) >= 1
    c = claims[0]
    assert c["thread_id"] == thread_id
    assert "shared-buffers" in c["chunk_id"]
    assert len(c["chunk_sha256"]) > 10
    assert c["verifier_verdict"] == "PASS"
    assert c["verifier_code"] == "VERIFIED"


# --- 4. Test Secrets Redaction in Audit Ledger ---

def test_secrets_never_stored_in_audit_ledger():
    """
    Day 5 Req 3: Ensure API keys, tokens, and passwords are never persisted.
    """
    thread_id = f"test-secrets-{uuid.uuid4().hex[:6]}"
    turn_uuid = str(uuid.uuid4())

    record_turn(
        turn_id=turn_uuid,
        thread_id=thread_id,
        turn_number=1,
        target_version="16",
        query="Query with fake secrets in details",
        draft_answer="Clean answer",
    )

    # Attempt to write sensitive fields into step_ledger
    leak_attempt = {
        "api_key": "AIzaSyFakeKey12345",
        "gemini_api_key": "AIzaSecretKey999",
        "password": "super_secret_password",
        "normal_field": "public_data",
    }
    record_step(
        turn_id=turn_uuid,
        thread_id=thread_id,
        step_number=1,
        node="test_node",
        action="test_action",
        status="success",
        decided_by="rule",
        latency_ms=5.0,
        details=leak_attempt,
    )

    steps = fetch_steps(thread_id, turn_id=turn_uuid)
    assert len(steps) == 1
    stored_details = steps[0]["details"]

    assert stored_details.get("normal_field") == "public_data"
    assert stored_details.get("api_key") == "[REDACTED]"
    assert stored_details.get("gemini_api_key") == "[REDACTED]"
    assert stored_details.get("password") == "[REDACTED]"
    assert "AIzaSyFakeKey12345" not in str(stored_details)
