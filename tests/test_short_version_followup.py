"""
test_short_version_followup.py: Synthetic unit tests for Item 2:
  - Treat short, setting-less questions containing a version or starting with
    and/in/also/how about/what about as follow-ups ("And in 15?", "In 16?", "How about 14?", "Same in 15?").
  - Keep non-follow-ups out ("In 2024, who won the match?").
  - Verify safe_substitute_pronoun handles these short forms into clean target questions.
"""

import pytest
from evidenceflow.memory.graph_client import is_follow_up_query, safe_substitute_pronoun


@pytest.mark.parametrize(
    "query, expected_is_followup",
    [
        ("And in 15?", True),
        ("In 16?", True),
        ("How about 14?", True),
        ("Same in 15?", True),
        ("Also in 15?", True),
        ("What about 14?", True),
        ("In 2024, who won the match?", False),
        ("In France, who is the president?", False),
        ("What is the weather in London?", False),
    ],
)
def test_short_version_queries_followup_detection(query, expected_is_followup):
    """Verify short version follow-up queries are classified correctly while non-follow-ups are excluded."""
    actual = is_follow_up_query(query)
    assert actual is expected_is_followup, f"Failed for query: {query}"


def test_short_version_queries_rewriting():
    """Verify safe_substitute_pronoun generates coherent technical questions for short version queries."""
    param = "checkpoint_timeout"

    # "And in 15?" -> What is checkpoint_timeout in PostgreSQL 15?
    rewritten_and = safe_substitute_pronoun("And in 15?", param, target_version="15")
    assert rewritten_and is not None
    assert param in rewritten_and
    assert "15" in rewritten_and

    # "In 16?" -> What is checkpoint_timeout in PostgreSQL 16?
    rewritten_in = safe_substitute_pronoun("In 16?", param, target_version="16")
    assert rewritten_in is not None
    assert param in rewritten_in
    assert "16" in rewritten_in

    # "How about 14?" -> What is checkpoint_timeout in PostgreSQL 14?
    rewritten_how = safe_substitute_pronoun("How about 14?", param, target_version="14")
    assert rewritten_how is not None
    assert param in rewritten_how
    assert "14" in rewritten_how

    # "Same in 15?" -> Is checkpoint_timeout the same in PostgreSQL 15?
    rewritten_same = safe_substitute_pronoun("Same in 15?", param, target_version="15")
    assert rewritten_same is not None
    assert param in rewritten_same
    assert "15" in rewritten_same
