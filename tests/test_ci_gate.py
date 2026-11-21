"""
test_ci_gate.py: Unit tests for the CI Metric Regression Gate.
Verifies threshold enforcement for Groundedness, Recall@5, Cache Integrity,
Security Abstention Accuracy, and the Break It on Purpose CLI exit-code gate.
"""

import subprocess
import sys
import pytest
from unittest.mock import patch

from evidenceflow.eval.ci_gate import (
    run_ci_gate,
    CIGateReport,
    DEFAULT_MIN_GROUNDEDNESS,
    DEFAULT_MIN_RECALL_AT_5,
    DEFAULT_MAX_CACHE_FALSE_HITS,
    DEFAULT_MIN_ABSTENTION_ACCURACY,
)


def test_ci_gate_all_thresholds_pass():
    """
    Verifies that under production conditions, all quality regression gates pass
    and return an overall_passed status of True in < 3 seconds.
    """
    report = run_ci_gate(
        min_groundedness=DEFAULT_MIN_GROUNDEDNESS,
        min_recall_at_5=DEFAULT_MIN_RECALL_AT_5,
        max_cache_false_hits=DEFAULT_MAX_CACHE_FALSE_HITS,
        min_abstention_accuracy=DEFAULT_MIN_ABSTENTION_ACCURACY,
    )

    # Invariants under test:
    assert report.overall_passed is True
    assert report.evaluation_duration_s < 5.0

    # Individual metric assertions:
    for m in report.metrics:
        assert m.passed is True, f"Metric '{m.name}' failed gate: observed {m.observed}% vs threshold {m.threshold}%"


def test_ci_gate_rejects_groundedness_regression():
    """Verifies that Groundedness dropping below minimum threshold triggers an overall FAIL."""
    report = run_ci_gate(
        min_groundedness=1.05,  # Impossible threshold (105%)
        min_recall_at_5=DEFAULT_MIN_RECALL_AT_5,
        max_cache_false_hits=DEFAULT_MAX_CACHE_FALSE_HITS,
    )

    assert report.overall_passed is False
    grounded_metric = next(m for m in report.metrics if "Groundedness" in m.name)
    assert grounded_metric.passed is False


def test_ci_gate_rejects_recall_regression():
    """Verifies that Recall@5 dropping below minimum threshold triggers an overall FAIL."""
    report = run_ci_gate(
        min_groundedness=DEFAULT_MIN_GROUNDEDNESS,
        min_recall_at_5=0.999,  # Impossible threshold (99.9%)
        max_cache_false_hits=DEFAULT_MAX_CACHE_FALSE_HITS,
    )

    assert report.overall_passed is False
    recall_metric = next(m for m in report.metrics if "Recall@5" in m.name)
    assert recall_metric.passed is False


def test_ci_gate_rejects_cache_false_hits():
    """Verifies that non-zero cache false hits trigger an immediate regression FAIL."""
    with patch("evidenceflow.eval.ci_gate.evaluate_cache_false_hit_rate", return_value=0.25):
        report = run_ci_gate(
            min_groundedness=DEFAULT_MIN_GROUNDEDNESS,
            min_recall_at_5=DEFAULT_MIN_RECALL_AT_5,
            max_cache_false_hits=0.0,
        )

    assert report.overall_passed is False
    cache_metric = next(m for m in report.metrics if "Cache False-Hit Rate" in m.name)
    assert cache_metric.passed is False
    assert cache_metric.observed == 25.0


def test_break_it_on_purpose_ci_gate_cli_exit_code_1():
    """
    Break It on Purpose (Day 18 Section 6):
    Executes ci_gate via CLI with --break-it flag.
    Verifies that:
      1. The process exits with code 1 (blocking pull request merge).
      2. The diagnostic output flags [FAIL *REGRESSION*].
      3. Output explicitly states QUALITY REGRESSION DETECTED - BUILD FAILED.
    """
    cmd = [sys.executable, "-m", "evidenceflow.eval.ci_gate", "--break-it"]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Invariants:
    assert proc.returncode == 1, f"Expected exit code 1 on regression, got {proc.returncode}"
    output = proc.stdout + proc.stderr
    assert "QUALITY REGRESSION DETECTED - BUILD FAILED" in output
    assert "[FAIL *REGRESSION*]" in output
    assert "Exit Code: 1 (Merge Blocked" in output
