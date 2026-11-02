"""
Synthetic unit tests for possessive pronoun rules and multi-entity plural resolution.
Verifies requirement (e) and (h) under synthetic isolated parameters.
NOTE: Uses strictly synthetic parameters (do not use h/f benchmark cases).
"""

import pytest
import uuid
from evidenceflow.memory.corpus_settings import set_corpus_settings_override
from evidenceflow.memory.init_neo4j import drop_graph
from evidenceflow.memory.graph_client import (
    is_follow_up_query,
    extract_all_explicit_settings,
    record_turn,
    resolve_entity_for_turn,
)


@pytest.fixture(autouse=True)
def setup_synthetic_env():
    """Configures synthetic parameter names for clean test isolation."""
    synthetic_settings = {
        "synth_worker_count",
        "synth_max_tasks",
        "synth_buffer_cache",
    }
    set_corpus_settings_override(synthetic_settings)
    yield
    set_corpus_settings_override(None)


def test_possessive_pronoun_detection():
    """Requirement (e): possessive forms (its, their) must trigger follow-up detection."""
    assert is_follow_up_query("What is its default value?") is True
    assert is_follow_up_query("How do their limits compare?") is True
    assert is_follow_up_query("What about their configuration in v16?") is True


def test_possessive_pronoun_false_trigger_guard():
    """Requirement (e): non-followup uses of 'its' in documentation queries must not trigger."""
    assert is_follow_up_query("Where in its documentation is memory discussed?") is False
    assert is_follow_up_query("Explain synthetic engine and its source code.") is False


def test_multi_entity_extraction_turn1():
    """Requirement (h): extracts all settings from turn 1, not only the first one."""
    q = "What are the defaults of synth_worker_count and synth_max_tasks in v16?"
    settings = extract_all_explicit_settings(q)
    assert len(settings) == 2
    assert "synth_worker_count" in settings
    assert "synth_max_tasks" in settings


def test_plural_pronoun_clarification():
    """
    Requirement (h): Plural pronouns (those, these, both, them) after multiple entities
    must halt with ambiguous clarification rather than accidentally picking one.
    """
    thread_id = f"test_plural_{uuid.uuid4().hex[:6]}"
    drop_graph(thread_id)

    # Turn 1 mentions two synthetic parameters
    record_turn(
        turn_id=str(uuid.uuid4()),
        thread_id=thread_id,
        turn_number=1,
        question="What are synth_worker_count and synth_max_tasks?",
        version_scope="16",
        entities=[
            {"name": "synth_worker_count", "type": "ConfigParam"},
            {"name": "synth_max_tasks", "type": "ConfigParam"},
        ],
        is_abstained=False,
    )

    # Turn 2 with plural pronoun
    resolution = resolve_entity_for_turn(
        thread_id=thread_id,
        follow_up_query="Which of those requires a server restart?",
    )

    assert resolution["status"] == "ambiguous"
    assert resolution["reason"] == "multiple_entities_plural"
    assert "synth_worker_count" in resolution["clarification"]
    assert "synth_max_tasks" in resolution["clarification"]
    # Guarantee it did NOT resolve to a single setting
    assert resolution.get("entity_name") is None


def test_singular_pronoun_after_multiple_entities():
    """Requirement (h): Singular pronoun after multiple entities remains strictly ambiguous."""
    thread_id = f"test_singular_multi_{uuid.uuid4().hex[:6]}"
    drop_graph(thread_id)

    record_turn(
        turn_id=str(uuid.uuid4()),
        thread_id=thread_id,
        turn_number=1,
        question="What are synth_worker_count and synth_max_tasks?",
        version_scope="16",
        entities=[
            {"name": "synth_worker_count", "type": "ConfigParam"},
            {"name": "synth_max_tasks", "type": "ConfigParam"},
        ],
        is_abstained=False,
    )

    resolution = resolve_entity_for_turn(
        thread_id=thread_id,
        follow_up_query="Does it require a restart?",
    )

    assert resolution["status"] == "ambiguous"
    assert resolution["reason"] == "multiple_entities"
    assert resolution.get("entity_name") is None
