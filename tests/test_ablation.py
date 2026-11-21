"""
test_ablation.py: Unit tests for Tier 2 ablation study runner and statistical non-inferiority testing.
"""

from unittest.mock import MagicMock, patch
import pytest

from evidenceflow.eval.ablation_runner import (
    AblationCondition,
    calculate_cost_per_1k,
    calculate_percentiles,
    run_condition_evaluation,
)
from evidenceflow.eval.run_ablation import compute_non_inferiority_test


def test_percentile_calculation():
    """Calculates exact median (p50) and tail (p95) latency percentiles."""
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    pcts = calculate_percentiles(data, [50.0, 90.0, 95.0])
    assert pcts[50.0] == 55.0
    assert pcts[90.0] == 91.0
    assert pcts[95.0] == 95.5


def test_cost_calculation():
    """Calculates USD cost per 1,000 queries based on prompt and candidate tokens."""
    # 1,000 queries with 1,000 prompt tokens and 200 completion tokens each
    cost = calculate_cost_per_1k(
        total_prompt_tokens=1_000_000,
        total_completion_tokens=200_000,
        num_queries=1000,
    )
    # Expected: 1M prompt ($0.075) + 0.2M candidate (0.2 * $0.30 = $0.060) = $0.135 per 1k
    assert round(cost, 4) == 0.1350


def test_statistical_non_inferiority_pass():
    """Identical or within-margin distributions pass statistical non-inferiority."""
    baseline = [1.0] * 30
    treatment = [1.0] * 30

    stats = compute_non_inferiority_test(baseline, treatment, delta=0.02)
    assert stats["is_non_inferior"] is True
    assert stats["mean_diff"] == 0.0
    assert stats["ci_95_upper_bound"] < 0.02


def test_statistical_non_inferiority_rejects_quality_regression():
    """When treatment degrades quality beyond the margin delta, non-inferiority is rejected."""
    baseline = [1.0] * 30
    treatment = [0.8] * 30  # 20% degradation (far exceeding 2% margin)

    stats = compute_non_inferiority_test(baseline, treatment, delta=0.02)
    assert stats["is_non_inferior"] is False
    assert stats["mean_diff"] > 0.02


def test_break_it_on_purpose_impossible_delta():
    """
    Break It on Purpose (Roadmap Day 13 Section 6):
    Setting delta to an impossible margin (e.g. 0.0000001) causes natural variance
    to flag a non-inferiority warning.
    """
    # Realistic samples with tiny natural variance
    baseline = [1.0, 1.0, 1.0, 0.9, 1.0, 1.0]
    treatment = [1.0, 0.9, 1.0, 1.0, 1.0, 1.0]

    stats = compute_non_inferiority_test(baseline, treatment, delta=0.0000001)
    assert stats["is_non_inferior"] is False, "Expected non-inferiority rejection under impossible margin!"


@patch("evidenceflow.eval.ablation_runner.run_pipeline")
def test_ablation_condition_runner_fast(mock_pipeline):
    """Verifies pipeline toggling between Baseline (no cache) and +Cache."""
    # Mock pipeline state return
    mock_state = MagicMock()
    mock_state.l1_passed = True
    mock_state.is_abstained = False
    mock_state.draft_answer = "max_connections is 100 [^chunk1]."
    mock_state.retrieved_passages = [{"chunk_id": "chunk1", "sha256": "h1"}]
    mock_state.citations = [MagicMock(chunk_id="chunk1", is_verified=True)]
    mock_state.cache_hit = False
    mock_state.audit_trail = [{"prompt_tokens": 150, "candidate_tokens": 50}]
    mock_pipeline.return_value = mock_state

    sample_questions = [
        {"id": "q1", "query": "What is max_connections in v16?", "version": "16", "is_answerable": True},
    ]

    # Baseline run
    baseline_cond = AblationCondition("Baseline", enable_memory=False, enable_cache=False, description="Baseline")
    res_base = run_condition_evaluation(baseline_cond, sample_questions, repeats=2)

    assert res_base["total_executions"] == 2
    assert res_base["mean_groundedness"] == 1.0
    assert res_base["total_tokens"] == 400
