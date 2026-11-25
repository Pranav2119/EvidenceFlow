"""
test_ablation.py: Unit tests for Tier 2 ablation study runner and statistical non-inferiority testing.
"""

from unittest.mock import MagicMock, patch
import json

import pytest

from evidenceflow.eval.ablation_runner import (
    AblationCondition,
    calculate_cost_per_1k,
    calculate_percentiles,
    run_condition_evaluation,
)
import evidenceflow.eval.run_ablation as ablation
from evidenceflow.eval.run_ablation import (
    compute_non_inferiority_test,
    format_ablation_markdown_report,
)
from evidenceflow.state import Citation, EvidenceFlowState


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
    """A non-degenerate improvement can pass the explicitly approximate check."""
    baseline = [0.7, 0.9] * 20
    treatment = [0.8, 1.0] * 20

    stats = compute_non_inferiority_test(baseline, treatment, delta=0.02)
    assert stats["is_non_inferior"] is True
    assert stats["mean_diff"] == pytest.approx(-0.1)
    assert stats["ci_upper_bound"] < 0.02
    assert "APPROXIMATE" in stats["verdict"]
    assert "PROVEN" not in stats["verdict"]


@pytest.mark.parametrize(
    "baseline, treatment",
    [
        ([], []),
        ([1.0], [0.0]),
        ([1.0] * 30, [1.0] * 30),
    ],
)
def test_non_inferiority_is_inconclusive_without_estimable_uncertainty(baseline, treatment):
    stats = compute_non_inferiority_test(baseline, treatment)
    assert stats["is_non_inferior"] is False
    assert stats["ci_upper_bound"] is None
    assert stats["p_value"] is None
    assert "INCONCLUSIVE" in stats["verdict"]
    if baseline and treatment:
        expected_difference = sum(baseline) / len(baseline) - sum(treatment) / len(treatment)
        assert stats["mean_diff"] == pytest.approx(expected_difference)
    else:
        assert stats["mean_baseline"] is None
        assert stats["mean_treatment"] is None
        assert stats["mean_diff"] is None


def test_non_inferiority_uses_requested_confidence_level():
    baseline = [0.7, 0.9] * 20
    treatment = [0.8, 1.0] * 20
    standard = compute_non_inferiority_test(baseline, treatment, alpha=0.05)
    stricter = compute_non_inferiority_test(baseline, treatment, alpha=0.01)
    assert stricter["confidence_level"] == 0.99
    assert stricter["ci_upper_bound"] > standard["ci_upper_bound"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0},
        {"delta": float("nan")},
        {"baseline_samples": [1.0, float("nan")]},
    ],
)
def test_non_inferiority_rejects_invalid_inputs(kwargs):
    arguments = {"baseline_samples": [0.7, 0.9], "treatment_samples": [0.8, 1.0]}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        compute_non_inferiority_test(**arguments)


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


@pytest.mark.parametrize(
    "expected_fields, answer, accuracy",
    [
        ({"expected_value": "4MB"}, "The default is 4MB [^chunk1].", 100.0),
        ({"expected_values": ["4MB", "8MB"]}, "Values are 4MB and 8MB [^chunk1].", 100.0),
        ({"expected_values": ["4MB", "8MB"]}, "The default is 4MB [^chunk1].", 0.0),
    ],
)
def test_ablation_scores_each_required_value(expected_fields, answer, accuracy):
    state = EvidenceFlowState(
        query="Synthetic memory question",
        draft_answer=answer,
        retrieved_passages=[{"chunk_id": "chunk1"}],
        citations=[Citation(chunk_id="chunk1", doc_name="doc", section="Memory", quote="", is_verified=True)],
        l1_passed=True,
    )
    question = {"id": "synthetic", "query": state.query, "version": "16", **expected_fields}
    condition = AblationCondition("Baseline", False, False, "Synthetic baseline")
    with (
        patch("evidenceflow.eval.ablation_runner.run_pipeline", return_value=state),
        patch("evidenceflow.eval.ablation_runner.set_cache_manager"),
    ):
        result = run_condition_evaluation(condition, [question], repeats=1)
    assert result["accuracy_pct"] == accuracy


@pytest.fixture
def condition_result():
    return {
        "condition": "Baseline",
        "description": "Synthetic",
        "total_executions": 1,
        "accuracy_pct": 100.0,
        "mean_groundedness": 1.0,
        "p50_latency_ms": 1.0,
        "p95_latency_ms": 1.0,
        "cached_p50_latency_ms": 0.0,
        "tokens_per_query": 0.0,
        "cost_per_1k_usd": 0.0,
        "cache_hit_rate_pct": 0.0,
        "groundedness_distribution": [1.0],
    }


def test_inconclusive_ablation_report_does_not_format_missing_statistics_as_numbers(condition_result):
    stats = compute_non_inferiority_test([], [])
    report = format_ablation_markdown_report(
        [condition_result, {**condition_result, "condition": "+Both"}],
        stats,
        git_commit="synthetic",
        dataset_name="synthetic data",
        repeats=1,
    )
    assert "INCONCLUSIVE" in report
    assert "N/A" in report
    assert "PROVEN NON-INFERIOR" not in report


@pytest.mark.parametrize("break_it", [False, True])
def test_ablation_cli_reports_inconclusive_without_success_exit(
    tmp_path, monkeypatch, capsys, condition_result, break_it,
):
    dataset = tmp_path / "synthetic.json"
    dataset.write_text(json.dumps({"version": "synthetic-v1", "questions": [{"id": "synthetic", "query": "Synthetic question"}]}), encoding="utf-8")
    report = tmp_path / "report.md"
    (tmp_path / "data" / "benchmarks").mkdir(parents=True)
    monkeypatch.setattr(ablation, "__file__", str(tmp_path / "src" / "evidenceflow" / "eval" / "run_ablation.py"))
    monkeypatch.setattr(ablation.sys, "argv", [
        "run_ablation", "--dataset", str(dataset), "--output", str(report),
        "--repeats", "1", "--subset", "1",
    ] + (["--break-it"] if break_it else []))
    monkeypatch.setattr(
        ablation, "run_condition_evaluation",
        lambda condition, **kwargs: {**condition_result, "condition": condition.name},
    )
    monkeypatch.setattr(ablation.subprocess, "check_output", lambda *args, **kwargs: "synthetic")
    monkeypatch.setattr(ablation, "close_connection_pool", MagicMock())
    monkeypatch.setattr(ablation, "close_neo4j_driver", MagicMock())

    with pytest.raises(SystemExit) as exc:
        ablation.main()

    assert exc.value.code == 1
    assert "INCONCLUSIVE" in report.read_text(encoding="utf-8")
    assert "PROOF COMPLETED SUCCESSFULLY" not in capsys.readouterr().out
    saved = json.loads((tmp_path / "data" / "benchmarks" / "ablation_tier2_results.json").read_text(encoding="utf-8"))
    assert saved["dataset_version"] == "synthetic-v1"
    assert len(saved["dataset_sha256"]) == 64
    assert saved["dataset_sha256"] in report.read_text(encoding="utf-8")
