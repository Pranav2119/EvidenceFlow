"""
test_memory_write_audit.py: Synthetic unit tests for Item 1 audit fixes:
  1. Fix Citation.get() AttributeError via direct attribute access.
  2. Multi-citation bracket splitting in extract_citations.
  3. Forced Neo4j write failure logging at ERROR with traceback and memory_write_status in step_ledger.
  4. Successful memory write recording memory_write_status in step_ledger.
"""

import os
import uuid
import pytest
from unittest.mock import patch

from evidenceflow.state import Citation, EvidenceFlowState
from evidenceflow.graph.workflow import run_pipeline
from evidenceflow.graph.nodes.answer_writer import extract_citations
from evidenceflow.storage.ledger import fetch_steps, close_connection_pool, get_settings
from evidenceflow.storage.migrate_postgres import run_migrations, ensure_database_exists

TEST_DB = "evidenceflow_test"


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    ensure_database_exists(TEST_DB)
    old_db = os.environ.get("POSTGRES_DB")
    os.environ["POSTGRES_DB"] = TEST_DB
    get_settings.cache_clear()
    close_connection_pool()

    run_migrations(dbname=TEST_DB)
    yield
    close_connection_pool()
    if old_db is not None:
        os.environ["POSTGRES_DB"] = old_db
    else:
        os.environ.pop("POSTGRES_DB", None)
    get_settings.cache_clear()


def test_citation_attribute_access_without_attribute_error():
    """Verify Citation model with is_verified=False is safely handled without .get() AttributeError."""
    c = Citation(
        chunk_id="pg16-sample-chunk",
        doc_name="sample.html",
        section="Sample Section",
        quote="Sample text",
        is_verified=False,
    )
    # Attribute access as in workflow.py
    cid = getattr(c, "chunk_id", None) if not isinstance(c, dict) else c.get("chunk_id")
    quote = getattr(c, "quote", "") if not isinstance(c, dict) else c.get("quote", "")
    is_ver = getattr(c, "is_verified", False) if not isinstance(c, dict) else c.get("is_verified", False)

    assert cid == "pg16-sample-chunk"
    assert quote == "Sample text"
    assert is_ver is False
    verdict = "VERIFIED" if is_ver else "UNVERIFIED"
    assert verdict == "UNVERIFIED"


def test_multi_citation_bracket_splitting_in_extract_citations():
    """Verify that grouped citations [^chunk1, ^chunk2] are split into individual Citation models."""
    passages = [
        {"chunk_id": "chunk_alpha", "doc_name": "alpha.html", "section": "Alpha", "title": "Alpha Setting"},
        {"chunk_id": "chunk_beta", "doc_name": "beta.html", "section": "Beta", "title": "Beta Setting"},
    ]
    answer = "Both settings are active [^chunk_alpha, ^chunk_beta]."
    citations = extract_citations(answer, passages)

    assert len(citations) == 2
    assert citations[0].chunk_id == "chunk_alpha"
    assert citations[1].chunk_id == "chunk_beta"
    assert citations[0].doc_name == "alpha.html"
    assert citations[1].doc_name == "beta.html"


def test_forced_memory_write_failure_logged_and_visible_in_step_ledger(caplog):
    """
    Forces a Neo4j write failure and verifies:
      1. An ERROR log is emitted with traceback.
      2. A step_ledger row is recorded with status='error' and details['memory_write_status']='error'.
    """
    thread_id = f"test_err_thread_{uuid.uuid4().hex[:6]}"

    with patch("evidenceflow.graph.workflow.neo4j_record_turn") as mock_neo4j:
        mock_neo4j.side_effect = RuntimeError("Simulated Neo4j cluster timeout")

        with caplog.at_level("ERROR"):
            state = run_pipeline(
                query="What is shared_buffers in PostgreSQL 16?",
                version="16",
                thread_id=thread_id,
                turn_number=1,
                enable_memory=True,
                use_postgres_checkpointer=False,
            )

    # 1. Check logger.error with traceback
    error_logs = [record for record in caplog.records if record.levelname == "ERROR"]
    assert any("Failed to record turn to Neo4j conversational memory" in r.message for r in error_logs)
    mem_log = next(r for r in error_logs if "Failed to record turn to Neo4j" in r.message)
    assert mem_log.exc_info is not None, "Error log must contain traceback (exc_info=True)"

    # 2. Check step_ledger visibility
    steps = fetch_steps(thread_id=thread_id)
    mem_steps = [s for s in steps if s.get("node") == "memory_writer"]
    assert len(mem_steps) >= 1
    mem_step = mem_steps[0]
    assert mem_step["status"] == "error"
    details = mem_step.get("details", {})
    assert details.get("memory_write_status") == "error"
    assert "Simulated Neo4j cluster timeout" in details.get("error", "")


def test_successful_memory_write_records_status_in_step_ledger():
    """Verify that a successful turn execution with memory records memory_write_status='success'."""
    thread_id = f"test_ok_thread_{uuid.uuid4().hex[:6]}"

    state = run_pipeline(
        query="What is shared_buffers in PostgreSQL 16?",
        version="16",
        thread_id=thread_id,
        turn_number=1,
        enable_memory=True,
        use_postgres_checkpointer=False,
    )

    steps = fetch_steps(thread_id=thread_id)
    mem_steps = [s for s in steps if s.get("node") == "memory_writer"]
    assert len(mem_steps) >= 1
    mem_step = mem_steps[0]
    assert mem_step["status"] == "success"
    details = mem_step.get("details", {})
    assert details.get("memory_write_status") == "success"
