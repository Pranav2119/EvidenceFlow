"""
test_offline_writer_defaults.py: Synthetic unit tests for offline deterministic writer.
Verifies:
  1. Question about default matches "The default is...", "The default value is...", "It defaults to...", "(which is the default)".
  2. Plain "What is X" returns the definition sentence without abstention.
  3. Absent asked attribute (e.g. restart behavior not in text) abstains with NO_SUBSTANTIVE_EVIDENCE.
  4. Numbered breadcrumbs (e.g. 20.4., E.16.2.) are stripped from answer text.
"""

import pytest
from evidenceflow.graph.nodes.answer_writer import deterministic_fallback_answer, FallbackAnswer


def test_offline_writer_matches_default_sentence():
    passages = [
        {
            "chunk_id": "pg16-runtime-config-resource-work-mem",
            "section": "20.4. Resource Consumption > 20.4.1. Memory",
            "title": "work_mem ( integer )",
            "content": (
                "Section: 20.4. Resource Consumption > 20.4.1. Memory\n"
                "Parameter: work_mem ( integer )\n\n"
                "Sets the base maximum amount of memory to be used by a query operation before writing to temporary disk files. "
                "The default value is four megabytes ( 4MB ). "
                "Sort operations are used for ORDER BY."
            ),
        }
    ]
    ans, code = deterministic_fallback_answer("What is the default value of work_mem in PostgreSQL 16?", "16", passages)
    assert code is None, "Should not abstain on valid default question"
    assert "The default value is four megabytes ( 4MB )" in ans
    assert "20.4." not in ans, "Numbered breadcrumb must be stripped from answer text"
    assert "Resource Consumption > Memory" in ans


def test_offline_writer_plain_question_gets_definition():
    passages = [
        {
            "chunk_id": "pg16-runtime-config-resource-work-mem",
            "section": "20.4. Resource Consumption > 20.4.1. Memory",
            "title": "work_mem ( integer )",
            "content": (
                "Section: 20.4. Resource Consumption > 20.4.1. Memory\n"
                "Parameter: work_mem ( integer )\n\n"
                "Sets the base maximum amount of memory to be used by a query operation before writing to temporary disk files. "
                "The default value is four megabytes ( 4MB ). "
                "Sort operations are used for ORDER BY."
            ),
        }
    ]
    ans, code = deterministic_fallback_answer("What is work_mem in PostgreSQL 16?", "16", passages)
    assert code is None, "Plain question must not abstain"
    assert "Sets the base maximum amount of memory" in ans


def test_offline_writer_abstains_on_missing_asked_attribute():
    passages = [
        {
            "chunk_id": "pg16-runtime-config-resource-work-mem",
            "section": "20.4. Resource Consumption > 20.4.1. Memory",
            "title": "work_mem ( integer )",
            "content": (
                "Section: 20.4. Resource Consumption > 20.4.1. Memory\n"
                "Parameter: work_mem ( integer )\n\n"
                "Sets the base maximum amount of memory to be used by a query operation before writing to temporary disk files. "
                "The default value is four megabytes ( 4MB )."
            ),
        }
    ]
    # Passage does not mention restart / postmaster
    ans, code = deterministic_fallback_answer("Does changing work_mem require a server restart?", "16", passages)
    assert code == "NO_SUBSTANTIVE_EVIDENCE", "Must abstain when asked attribute is absent from text"


def test_offline_writer_parenthesized_default_matching():
    passages = [
        {
            "chunk_id": "pg16-runtime-config-autovacuum-autovacuum-vacuum-cost-limit",
            "section": "20.10. Automatic Vacuuming",
            "title": "autovacuum_vacuum_cost_limit ( integer )",
            "content": (
                "Section: 20.10. Automatic Vacuuming\n"
                "Parameter: autovacuum_vacuum_cost_limit ( integer )\n\n"
                "Specifies the cost limit value that will be used in automatic VACUUM operations. "
                "If -1 is specified (which is the default), the regular vacuum_cost_limit value will be used."
            ),
        }
    ]
    ans, code = deterministic_fallback_answer("What is the default value of autovacuum_vacuum_cost_limit in PostgreSQL 16?", "16", passages)
    assert code is None
    assert "If -1 is specified (which is the default)" in ans
    assert "20.10." not in ans
    assert "Automatic Vacuuming" in ans
