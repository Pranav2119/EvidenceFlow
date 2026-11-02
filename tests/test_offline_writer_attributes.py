"""
test_offline_writer_attributes.py: Synthetic unit tests for Item 6:
  - Choose the sentence that matches the asked attribute (restart, change, default, per-table).
  - If none matches, abstain with a clear code (ATTRIBUTE_NOT_SPECIFIED).
  - Do not count a generic definition as an answer when an attribute is asked.
"""

import pytest
from evidenceflow.graph.nodes.answer_writer import deterministic_fallback_answer


@pytest.fixture
def sample_passages():
    return [
        {
            "chunk_id": "pg16-runtime-config-wal-checkpoint-timeout",
            "doc_name": "runtime-config-wal.html",
            "section": "Write-Ahead Log > Checkpoints",
            "title": "checkpoint_timeout (integer)",
            "content": (
                "Section: Write-Ahead Log\n"
                "Parameter: checkpoint_timeout (integer)\n"
                "Specifies the maximum time between automatic WAL checkpoints. "
                "The default is 5 minutes (5min). "
                "This parameter can be changed in postgresql.conf without server restart."
            ),
        }
    ]


@pytest.fixture
def passages_without_per_table():
    return [
        {
            "chunk_id": "pg16-runtime-config-connection-max-connections",
            "doc_name": "runtime-config-connection.html",
            "section": "Connections and Authentication > Connection Settings",
            "title": "max_connections (integer)",
            "content": (
                "Section: Connections and Authentication\n"
                "Parameter: max_connections (integer)\n"
                "Determines the maximum number of concurrent connections to the database server. "
                "The default is typically 100 connections. "
                "This parameter can only be set at server start."
            ),
        }
    ]


def test_offline_writer_selects_default_sentence(sample_passages):
    """Verify asked default attribute selects the default statement."""
    answer = deterministic_fallback_answer("What is checkpoint_timeout's default?", "16", sample_passages)
    text, code = answer
    assert code is None
    assert "5 minutes" in text
    assert "[^pg16-runtime-config-wal-checkpoint-timeout]" in text


def test_offline_writer_selects_restart_sentence(sample_passages):
    """Verify asked restart attribute selects the restart statement."""
    answer = deterministic_fallback_answer("Does checkpoint_timeout require a restart?", "16", sample_passages)
    text, code = answer
    assert code is None
    assert "without server restart" in text
    assert "[^pg16-runtime-config-wal-checkpoint-timeout]" in text


def test_offline_writer_abstains_when_attribute_missing_no_generic_fallback(passages_without_per_table):
    """Verify writer abstains when asked attribute (per-table) is missing and does not fall back to generic definition."""
    answer = deterministic_fallback_answer("Can I set max_connections per-table?", "16", passages_without_per_table)
    text, code = answer
    # Must abstain with clear code
    assert code == "NO_SUBSTANTIVE_EVIDENCE"
    assert "does not specify per-table behavior" in text
    # Must NOT have returned the generic definition ("Determines the maximum number...")
    assert "Determines the maximum number" not in text


def test_offline_writer_allows_generic_definition_when_no_attribute_asked(passages_without_per_table):
    """Verify writer returns definition sentence when no specific attribute is requested."""
    answer = deterministic_fallback_answer("What is max_connections?", "16", passages_without_per_table)
    text, code = answer
    assert code is None
    assert "Determines the maximum number of concurrent connections" in text
