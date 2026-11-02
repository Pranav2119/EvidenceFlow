"""
test_offline_writer_substantive.py: Synthetic unit tests for Item i.
Verifies that the offline deterministic answer writer:
  1. Extracts substantive sentence(s) that answer the question (e.g. default sentence).
  2. Never outputs only a section header.
  3. Fails closed and abstains with clear code 'NO_SUBSTANTIVE_EVIDENCE' when
     evidence cannot answer the specific query.
Uses 100% synthetic parameter names and text (zero h/f benchmark cases).
"""

import pytest
from evidenceflow.state import EvidenceFlowState
from evidenceflow.graph.nodes.answer_writer import (
    deterministic_fallback_answer,
    answer_writer_node,
)
from evidenceflow.memory.corpus_settings import set_corpus_settings_override


@pytest.fixture(autouse=True)
def setup_synthetic_corpus():
    synthetic_settings = {
        "synthetic_flush_delay",
        "synthetic_obsolete_knob",
        "synthetic_empty_knob",
        "synthetic_knob_no_default",
        "synthetic_buffer_size",
    }
    set_corpus_settings_override(synthetic_settings)
    yield
    set_corpus_settings_override(None)


def test_offline_writer_extracts_default_sentence_not_header():
    """Rule i: Writer outputs the sentence with the default, never just a section header."""
    passages = [{
        "chunk_id": "pg16-syn-flush-delay",
        "section": "12.3. Data Buffers > 12.3.4. Flushing",
        "content": (
            "Section: 12.3. Data Buffers > 12.3.4. Flushing\n"
            "Parameter: synthetic_flush_delay ( integer )\n\n"
            "Controls the delay in milliseconds before flushing pending data buffers. "
            "Setting this to zero forces synchronous write operations. "
            "The default is 500ms."
        )
    }]
    query = "What is the default of synthetic_flush_delay in PostgreSQL 16?"
    ans, code = deterministic_fallback_answer(query, "16", passages)

    assert code is None
    assert "500ms" in ans
    assert "The default is 500ms" in ans
    assert "[^pg16-syn-flush-delay]" in ans
    # Crucial: verify that the answer body is NOT just the section header
    assert not ans.endswith("12.3.4. Flushing [^pg16-syn-flush-delay]")


def test_offline_writer_extracts_removal_sentence():
    """Rule i: Writer outputs the sentence stating removal when asked about removal."""
    passages = [{
        "chunk_id": "pg16-release-notes",
        "section": "E.4. Release Notes",
        "content": (
            "Section: E.4. Release Notes > E.4.1. Migration\n\n"
            "Remove obsolete parameter synthetic_obsolete_knob. "
            "Applications should transition to dynamic buffer allocation."
        )
    }]
    query = "Was synthetic_obsolete_knob removed in PostgreSQL 16?"
    ans, code = deterministic_fallback_answer(query, "16", passages)

    assert code is None
    assert "Remove obsolete parameter synthetic_obsolete_knob" in ans
    assert "[^pg16-release-notes]" in ans


def test_offline_writer_abstains_on_header_only_chunk():
    """Rule i: If chunk contains only section headers and no substantive sentence, abstain with clear code."""
    passages = [{
        "chunk_id": "pg16-empty-header-only",
        "section": "99.1. Empty Category",
        "content": "Section: 99.1. Empty Category > 99.1.2. Null Subsection"
    }]
    query = "What is the default of synthetic_empty_knob in PostgreSQL 16?"
    ans, code = deterministic_fallback_answer(query, "16", passages)

    assert code == "NO_SUBSTANTIVE_EVIDENCE"
    assert "Insufficient evidence" in ans


def test_offline_writer_abstains_when_default_asked_but_missing():
    """Rule i: If user specifically asks for default but passage has no default statement, abstain with clear code."""
    passages = [{
        "chunk_id": "pg16-syn-no-default",
        "section": "5.1. General Configuration",
        "content": (
            "Section: 5.1. General Configuration\n"
            "Parameter: synthetic_knob_no_default ( string )\n\n"
            "Specifies the custom cluster identifier tag sent to monitoring sidecars."
        )
    }]
    query = "What is the default of synthetic_knob_no_default in PostgreSQL 16?"
    ans, code = deterministic_fallback_answer(query, "16", passages)

    assert code == "NO_SUBSTANTIVE_EVIDENCE"
    assert "Insufficient evidence" in ans


def test_offline_writer_general_definition_query():
    """Rule i: For general questions, extracts substantive parameter purpose/definition sentence."""
    passages = [{
        "chunk_id": "pg16-syn-buffer-size",
        "section": "4.1. Memory Management",
        "content": (
            "Section: 4.1. Memory Management\n"
            "Parameter: synthetic_buffer_size ( integer )\n\n"
            "Sets the amount of dedicated memory allocated for buffering operations. "
            "The default is 64MB."
        )
    }]
    query = "What does synthetic_buffer_size do in PostgreSQL 16?"
    ans, code = deterministic_fallback_answer(query, "16", passages)

    assert code is None
    assert "Sets the amount of dedicated memory allocated for buffering operations." in ans
    assert "[^pg16-syn-buffer-size]" in ans


def test_answer_writer_node_propagates_offline_abstention(monkeypatch):
    """Rule i: answer_writer_node properly sets is_abstained=True and abstention_code on fallback failure."""
    from evidenceflow.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")

    state = EvidenceFlowState(
        query="What is the default of synthetic_empty_knob in PostgreSQL 16?",
        target_version="16",
        retrieved_passages=[{
            "chunk_id": "pg16-empty-header",
            "section": "99.1. Empty Category",
            "content": "Section: 99.1. Empty Category > 99.1.2. Null Subsection",
        }],
    )
    res = answer_writer_node(state)
    assert res["is_abstained"] is True
    assert res["abstention_code"] == "NO_SUBSTANTIVE_EVIDENCE"
    assert res["synthesis_mode"] == "fallback"
