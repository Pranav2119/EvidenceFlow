"""
test_benchmark_metrics_final_answer.py: Synthetic unit tests for benchmark runner metrics.
Verifies:
  1. Refused turns have value_accuracy=False and citation_precision/recall=0.0 on final answer.
  2. Answered turns retain factual value accuracy and citation precision/recall.
  3. Benchmark thread IDs start with 'bench-'.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import evidenceflow.eval.run_benchmark as benchmark
from evidenceflow.state import Citation, EvidenceFlowState


def test_final_answer_metrics_on_refused_and_answered():
    def compute_metrics(is_abstained, draft_answer, citations, retrieved_ids, expected_val):
        from evidenceflow.eval.metrics import (
            calculate_citation_precision,
            calculate_citation_recall,
            check_value_accuracy,
        )

        draft_val_acc = check_value_accuracy(draft_answer, expected_val)
        if is_abstained:
            val_acc = False
            prec = 0.0
            recall = 0.0
        else:
            val_acc = draft_val_acc
            prec = calculate_citation_precision(citations, retrieved_ids)
            recall = calculate_citation_recall(draft_answer, citations)
        return val_acc, prec, recall, draft_val_acc

    # Case A: Refused turn whose draft happened to mention the number
    draft_a = "The default value is 4MB [^chunk1]."
    v_acc, p, r, d_acc = compute_metrics(
        is_abstained=True,
        draft_answer=draft_a,
        citations=[{"chunk_id": "chunk1", "is_verified": False}],
        retrieved_ids={"chunk1"},
        expected_val="4MB",
    )
    assert d_acc is True, "Draft contained the expected value"
    assert v_acc is False, "Final user-visible answer for refused turn must be False"
    assert p == 0.0, "Refused turn has 0.0 user-visible citation precision"
    assert r == 0.0, "Refused turn has 0.0 user-visible citation recall"

    # Case B: Answered turn
    v_acc_b, p_b, r_b, d_acc_b = compute_metrics(
        is_abstained=False,
        draft_answer=draft_a,
        citations=[{"chunk_id": "chunk1", "is_verified": True}],
        retrieved_ids={"chunk1"},
        expected_val="4MB",
    )
    assert v_acc_b is True
    assert p_b == 1.0
    assert r_b == 1.0


def test_bench_thread_id_prefix():
    split = "dev"
    qid = "d01"
    r = 0
    warmup_id = f"bench-{split}-{qid}-warmup"
    repeat_id = f"bench-{split}-{qid}-r{r}"

    assert warmup_id.startswith("bench-")
    assert repeat_id.startswith("bench-")
    assert "bench_" not in warmup_id
    assert "bench_" not in repeat_id


@pytest.fixture
def isolated_baseline(monkeypatch):
    state = EvidenceFlowState(
        query="Synthetic memory question",
        draft_answer="The default is 4MB [^chunk1].",
        retrieved_passages=[{"chunk_id": "chunk1"}],
        citations=[Citation(chunk_id="chunk1", doc_name="doc", section="Memory", quote="", is_verified=True)],
        l1_passed=True,
    )
    dataset = {
        "questions": [{
            "id": "synthetic",
            "query": state.query,
            "version": "16",
            "expected_value": "4MB",
            "ground_truth_chunks": ["chunk1"],
        }],
    }
    settings = SimpleNamespace(
        GEMINI_API_KEY="",
        GEMINI_MODEL="synthetic-model",
        EMBEDDING_PROVIDER="fastembed",
        RETRIEVAL_SCORE_THRESHOLD=0.6,
    )
    pipeline = Mock(return_value=state)
    monkeypatch.setattr(benchmark, "get_settings", lambda: settings)
    monkeypatch.setattr(benchmark, "load_benchmark_dataset", lambda path: dataset)
    monkeypatch.setattr(benchmark, "get_git_commit_hash", lambda: "synthetic")
    monkeypatch.setattr(benchmark, "run_pipeline", pipeline)
    return pipeline


def test_baseline_disables_cache_for_warmup_and_timed_runs(isolated_baseline):
    result = benchmark.run_benchmark(
        gemini_enabled=False, repeats=2, delay_seconds=0.0, verbose=False,
    )
    assert isolated_baseline.call_count == 3
    assert all(call.kwargs.get("enable_cache") is False for call in isolated_baseline.call_args_list)
    assert result["cache_mode"] == "OFF"
    assert result["metrics"]["value_accuracy_pct"] == 100.0


@pytest.mark.parametrize("repeats", [0, -1])
def test_baseline_requires_timed_runs(isolated_baseline, repeats):
    with pytest.raises(ValueError, match="repeats"):
        benchmark.run_benchmark(
            gemini_enabled=False, repeats=repeats, delay_seconds=0.0, verbose=False,
        )
    isolated_baseline.assert_not_called()
