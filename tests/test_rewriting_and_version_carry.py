"""
Synthetic unit tests for safe pronoun substitution rewriting and version carryover.
Verifies requirements (b) and (d) with synthetic test parameters.
NOTE: Uses strictly synthetic parameters (do not use h/f benchmark cases).
"""

import pytest
import uuid
from evidenceflow.memory.corpus_settings import set_corpus_settings_override
from evidenceflow.memory.init_neo4j import drop_graph
from evidenceflow.memory.graph_client import (
    safe_substitute_pronoun,
    record_turn,
    resolve_entity_for_turn,
)


@pytest.fixture(autouse=True)
def setup_synthetic_env():
    """Configures synthetic parameter names for clean test isolation."""
    synthetic_settings = {
        "synth_primary_setting",
        "synth_secondary_setting",
        "synth_wal_threshold",
    }
    set_corpus_settings_override(synthetic_settings)
    yield
    set_corpus_settings_override(None)


def test_safe_pronoun_substitution_preserves_question_grammar():
    """Requirement (b): substitutes pronoun with setting name while preserving intent."""
    q1 = "Does that setting require a restart?"
    rewritten1 = safe_substitute_pronoun(q1, "synth_primary_setting", "16")
    assert rewritten1 == "Does synth_primary_setting require a restart in PostgreSQL 16?"

    q2 = "What is its default in version 15?"
    rewritten2 = safe_substitute_pronoun(q2, "synth_wal_threshold", "15")
    assert rewritten2 == "What is synth_wal_threshold's default in version 15?"

    q3 = "What is the default value of this parameter?"
    rewritten3 = safe_substitute_pronoun(q3, "synth_secondary_setting", "14")
    assert rewritten3 == "What is the default value of synth_secondary_setting in PostgreSQL 14?"


def test_standalone_version_switch_substitution():
    """Requirement (b): standalone version switch questions rewrite cleanly."""
    rewritten = safe_substitute_pronoun("in v15?", "synth_primary_setting", "15")
    assert "synth_primary_setting" in rewritten
    assert "15" in rewritten


def test_version_carryover_when_turn2_has_none():
    """
    Requirement (d): Carry the version from turn 1 when turn 2 has none
    and does not name another.
    """
    thread_id = f"test_ver_carry_{uuid.uuid4().hex[:6]}"
    drop_graph(thread_id)

    # Turn 1 established version 14
    record_turn(
        turn_id=str(uuid.uuid4()),
        thread_id=thread_id,
        turn_number=1,
        question="What is synth_primary_setting in PostgreSQL 14?",
        version_scope="14",
        entities=[{"name": "synth_primary_setting", "type": "ConfigParam"}],
        is_abstained=False,
    )

    # Turn 2 has pronoun and NO version mentioned
    res_pronoun = resolve_entity_for_turn(
        thread_id=thread_id,
        follow_up_query="Does it require a restart?",
    )
    assert res_pronoun["status"] == "resolved"
    assert res_pronoun["target_version"] == "14"
    assert "synth_primary_setting" in res_pronoun["rewritten_query"]
    assert "14" in res_pronoun["rewritten_query"]

    # Turn 2 is a new explicit setting but with NO version flag
    res_explicit = resolve_entity_for_turn(
        thread_id=thread_id,
        follow_up_query="What is the default of synth_secondary_setting?",
    )
    assert res_explicit["status"] == "passthrough"
    assert res_explicit["entity_name"] == "synth_secondary_setting"
    # Target version carried over from turn 1 (14) rather than defaulting to None/16
    assert res_explicit["target_version"] == "14"


def test_explicit_version_in_turn2_overrides_turn1_version():
    """Requirement (d): Explicit version in turn 2 wins over turn 1 version."""
    thread_id = f"test_ver_override_{uuid.uuid4().hex[:6]}"
    drop_graph(thread_id)

    # Turn 1 established version 14
    record_turn(
        turn_id=str(uuid.uuid4()),
        thread_id=thread_id,
        turn_number=1,
        question="What is synth_primary_setting in PostgreSQL 14?",
        version_scope="14",
        entities=[{"name": "synth_primary_setting", "type": "ConfigParam"}],
        is_abstained=False,
    )

    # Turn 2 follow-up with explicit version 16
    res = resolve_entity_for_turn(
        thread_id=thread_id,
        follow_up_query="What is its default in version 16?",
    )
    assert res["status"] == "resolved"
    assert res["target_version"] == "16"
