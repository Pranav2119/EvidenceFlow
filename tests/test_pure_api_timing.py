"""
test_pure_api_timing.py: Synthetic unit tests for Item m.
Verifies that:
  1. Pure Gemini API call duration is timed independently of prompt building, pacing, and retries.
  2. Retries, sleeps, and total node latency are logged separately in step_ledger details.
Uses 100% synthetic mock responses and queries (zero benchmark cases).
"""

import time
from unittest.mock import MagicMock, patch
import pytest
from evidenceflow.state import EvidenceFlowState
from evidenceflow.graph.nodes.answer_writer import answer_writer_node
from evidenceflow.graph.audit_wrapper import audit_node_wrapper
from evidenceflow.config import get_settings


def test_pure_gemini_api_latency_recorded(monkeypatch):
    """Rule m: Times the pure Gemini API call and captures attempts and zero sleep in details."""
    settings = get_settings()
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "AIzaSyFakeKeyForTimingTest1234567")

    mock_response = MagicMock()
    mock_response.text = "Synthetic verified answer with [^pg16-syn-pure-call]."
    mock_response.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5, total_token_count=15)

    def slow_generate(*args, **kwargs):
        time.sleep(0.04)  # Simulate 40ms pure network call
        return mock_response

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = slow_generate

    with patch("evidenceflow.graph.nodes.answer_writer._get_writer_client", return_value=mock_client):
        state = EvidenceFlowState(
            query="What is synthetic_timing_knob in PostgreSQL 16?",
            target_version="16",
            retrieved_passages=[{
                "chunk_id": "pg16-syn-pure-call",
                "section": "9.9. Synthetic Section",
                "content": "synthetic_timing_knob is an integer knob with default 100.",
            }],
        )
        res = answer_writer_node(state)

    audit_entry = res["audit_trail"][-1]
    assert audit_entry["synthesis_mode"] == "llm"
    assert audit_entry["gemini_pure_api_latency_ms"] >= 35.0  # Captured ~40ms pure API call
    assert audit_entry["gemini_attempts"] == 1
    assert audit_entry["gemini_sleep_duration_ms"] == 0.0


def test_gemini_retries_and_sleeps_isolated_in_details(monkeypatch):
    """Rule m: Logs retries and sleeps separately without contaminating the pure API call metric."""
    settings = get_settings()
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "AIzaSyFakeKeyForTimingTest1234567")

    import httpx

    mock_response = MagicMock()
    mock_response.text = "Synthetic answer after retry [^pg16-syn-retry-chunk]."
    mock_response.usage_metadata = None

    call_count = 0
    def retry_then_succeed(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.HTTPError("503 Service Unavailable")
        return mock_response

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = retry_then_succeed

    # Mock time.sleep so test runs in sub-millisecond time while logging sleep duration
    sleep_calls = []
    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(time, "sleep", fake_sleep)

    with patch("evidenceflow.graph.nodes.answer_writer._get_writer_client", return_value=mock_client):
        state = EvidenceFlowState(
            query="What is synthetic_retry_param in PostgreSQL 16?",
            target_version="16",
            retrieved_passages=[{
                "chunk_id": "pg16-syn-retry-chunk",
                "section": "9.9. Synthetic Section",
                "content": "synthetic_retry_param is a boolean knob.",
            }],
        )
        res = answer_writer_node(state)

    audit_entry = res["audit_trail"][-1]
    assert audit_entry["synthesis_mode"] == "llm"
    assert audit_entry["gemini_attempts"] == 2
    assert len(sleep_calls) == 1
    assert sleep_calls[0] == 2.0  # 2.0s attempted backoff
    assert audit_entry["gemini_pure_api_latency_ms"] is not None


def test_audit_wrapper_isolates_pure_latency_for_llm(monkeypatch):
    """Rule m: audit_node_wrapper saves pure Gemini latency into step_ledger.latency_ms."""
    recorded_steps = []
    def fake_record_step(**kwargs):
        recorded_steps.append(kwargs)

    monkeypatch.setattr("evidenceflow.graph.audit_wrapper.record_step", fake_record_step)

    # Node returning LLM audit entry with pure API latency
    def mock_node(state):
        return {
            "draft_answer": "Synthetic grounded answer [^pg16-syn].",
            "is_abstained": False,
            "synthesis_mode": "llm",
            "audit_trail": [{
                "node": "answer_writer",
                "action": "answer_drafted",
                "synthesis_mode": "llm",
                "gemini_pure_api_latency_ms": 142.5,
                "gemini_attempts": 2,
                "gemini_sleep_duration_ms": 2000.0,
            }],
        }

    wrapped = audit_node_wrapper(mock_node, "answer_writer")
    state = EvidenceFlowState(
        query="What is synthetic_audit_test in PostgreSQL 16?",
        turn_id="turn-timing-123",
        thread_id="thread-timing-123",
    )

    wrapped(state)

    assert len(recorded_steps) == 1
    step = recorded_steps[0]
    assert step["decided_by"] == "llm"
    # Step latency is clean pure Gemini latency, NOT contaminated by node overhead
    assert step["latency_ms"] == 142.5
    # Retry and sleep details are preserved in details dict
    assert step["details"]["gemini_attempts"] == 2
    assert step["details"]["gemini_sleep_duration_ms"] == 2000.0
    assert "node_total_latency_ms" in step["details"]
