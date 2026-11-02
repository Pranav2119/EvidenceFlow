"""
test_version_word_followup.py: Synthetic unit tests for the 'version' keyword follow-up queries:
  - "What about version 16?"
  - "How about version 15?"
  - "And version 14?"
  - "What about 16?"
  - Control: "In version 2 of the contract, who signs?" is NOT rewritten.
"""

import pytest
from evidenceflow.memory.graph_client import is_follow_up_query, safe_substitute_pronoun, resolve_entity_for_turn


@pytest.mark.parametrize(
    "query, expected_ver",
    [
        ("What about version 16?", "16"),
        ("How about version 15?", "15"),
        ("And version 14?", "14"),
        ("What about 16?", "16"),
    ],
)
def test_version_word_queries_rewrite(query, expected_ver):
    """Verify short version follow-up queries containing 'version' are safely rewritten."""
    param = "log_checkpoints"
    rewritten = safe_substitute_pronoun(query, param, target_version=expected_ver)
    assert rewritten is not None
    assert param in rewritten
    assert expected_ver in rewritten
    assert "configuration and behavior" in rewritten


def test_version_word_followup_detection():
    """Verify 'version' follow-up queries are recognized as follow-up queries."""
    assert is_follow_up_query("What about version 16?") is True
    assert is_follow_up_query("How about version 15?") is True
    assert is_follow_up_query("And version 14?") is True
    assert is_follow_up_query("What about 16?") is True


def test_control_query_contract_not_rewritten():
    """Verify 'In version 2 of the contract, who signs?' is NOT recognized as a follow-up and NOT rewritten."""
    query = "In version 2 of the contract, who signs?"
    # 1. is_follow_up_query must be False due to off-corpus content words ('contract', 'signs')
    assert is_follow_up_query(query) is False

    # 2. safe_substitute_pronoun must return None (no pronouns, not a standalone version query)
    assert safe_substitute_pronoun(query, "log_checkpoints", target_version="2") is None
