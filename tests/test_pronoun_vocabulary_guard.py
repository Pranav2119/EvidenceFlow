"""
test_pronoun_vocabulary_guard.py: Synthetic unit tests for Item 4:
  - A pronoun follow-up must be shaped like a question about a setting.
  - If turn 2 has content words that are not in the corpus vocabulary (e.g. joke, funny, capital),
    do not rewrite (passthrough).
  - Tests in BOTH directions:
    * "Tell me a joke about servers. Is it funny?" -> passthrough
    * "What is the capital of France? Is it Paris?" -> passthrough
    * "Can I change it per table?" -> resolves
    * "Does it need a restart?" -> resolves
"""

import uuid
import pytest
from evidenceflow.memory.init_neo4j import init_neo4j_schema, drop_graph
from evidenceflow.memory.graph_client import (
    record_turn,
    resolve_entity_for_turn,
    is_follow_up_query,
)
from evidenceflow.memory.corpus_settings import has_off_corpus_content_words


@pytest.mark.parametrize(
    "query, expected_is_followup, expected_off_corpus",
    [
        ("Tell me a joke about servers. Is it funny?", False, True),
        ("What is the capital of France? Is it Paris?", False, True),
        ("Can I change it per table?", True, False),
        ("Does it need a restart?", True, False),
        ("What happens if I set it to 0?", True, False),
        ("Can I configure it at runtime?", True, False),
    ],
)
def test_pronoun_vocabulary_detection_both_directions(query, expected_is_followup, expected_off_corpus):
    """Verify follow-up detection distinguishes off-corpus content words from valid setting queries."""
    assert has_off_corpus_content_words(query) is expected_off_corpus
    assert is_follow_up_query(query) is expected_is_followup


def test_resolver_off_corpus_query_passes_through():
    """Verify off-corpus queries like jokes pass through untouched and do not resolve to prior setting."""
    init_neo4j_schema()
    thread_id = f"test-thread-off-corpus-{uuid.uuid4().hex[:8]}"
    turn_1_id = f"turn1_{uuid.uuid4().hex[:8]}"

    try:
        # Seed Turn 1 with setting "log_connections"
        record_turn(
            turn_id=turn_1_id,
            thread_id=thread_id,
            turn_number=1,
            question="What is log_connections in PostgreSQL 16?",
            version_scope="16",
            entities=[{"name": "log_connections", "type": "SETTING"}],
            claims=[{"id": f"{turn_1_id}_c1", "text": "log_connections is off by default.", "verdict": "VERIFIED"}],
            is_abstained=False,
        )

        # Off-corpus Turn 2 containing 'joke' and 'funny'
        off_query = "Tell me a joke about servers. Is it funny?"
        res = resolve_entity_for_turn(thread_id, off_query)

        assert res["status"] == "passthrough"
        assert res["entity_name"] is None
        assert res["rewritten_query"] == off_query
        assert "log_connections" not in res["rewritten_query"]

        # Valid setting query Turn 2 resolves
        valid_query = "Can I change it per table?"
        res_valid = resolve_entity_for_turn(thread_id, valid_query)

        assert res_valid["status"] == "resolved"
        assert res_valid["entity_name"] == "log_connections"
        assert "log_connections" in res_valid["rewritten_query"]
        assert "table" in res_valid["rewritten_query"]

        # Another valid setting query Turn 2 resolves
        valid_query_2 = "Does it need a restart?"
        res_valid_2 = resolve_entity_for_turn(thread_id, valid_query_2)

        assert res_valid_2["status"] == "resolved"
        assert res_valid_2["entity_name"] == "log_connections"
        assert "log_connections" in res_valid_2["rewritten_query"]
        assert "restart" in res_valid_2["rewritten_query"]

    finally:
        drop_graph(thread_id=thread_id)
