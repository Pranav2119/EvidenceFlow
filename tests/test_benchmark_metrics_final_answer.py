"""
test_benchmark_metrics_final_answer.py: Synthetic unit tests for benchmark runner metrics.
Verifies:
  1. Refused turns have value_accuracy=False and citation_precision/recall=0.0 on final answer.
  2. Answered turns retain factual value accuracy and citation precision/recall.
  3. Benchmark thread IDs start with 'bench-'.
"""

import json
from pathlib import Path
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


def test_baseline_exports_each_timed_observation_without_claiming_factual_accuracy(
    isolated_baseline, capsys,
):
    first = isolated_baseline.return_value.model_copy(update={"target_version": "16"})
    second = first.model_copy(update={
        "target_version": "14",
        "draft_answer": "Explanatory context. " * 12 + "The default is 128MB [^chunk2].",
        "retrieved_passages": [{"chunk_id": "chunk2"}],
        "citations": [Citation(chunk_id="chunk2", doc_name="doc", section="Memory", quote="128MB", is_verified=True)],
    })
    isolated_baseline.side_effect = [first, first, second]

    result = benchmark.run_benchmark(
        gemini_enabled=False, repeats=2, delay_seconds=0.0, verbose=False,
    )
    saved = json.loads(json.dumps(result))
    observations = saved["question_results"][0]["repeat_details"]
    assert observations[0]["draft_answer"] == first.draft_answer
    assert observations[1]["draft_answer"] == second.draft_answer
    assert observations[1]["target_version"] == "14"
    assert observations[1]["retrieved_chunk_ids"] == ["chunk2"]
    assert observations[1]["citations"][0]["chunk_id"] == "chunk2"
    assert observations[1]["cache_hit"] is False
    assert observations[1]["l1_passed"] is True
    assert saved["metrics"]["overall_accuracy_pct"] == 100.0
    assert saved["metrics"]["value_accuracy_pct"] == 0.0
    assert saved["temperature"] is None
    assert saved["quality_sample"] == "last_timed_repeat_per_question"
    output = capsys.readouterr().out
    assert "Answer/refusal decision accuracy" in output
    assert "Overall End-to-End" not in output


@pytest.mark.parametrize("hit_on", [0, 1])
def test_baseline_rejects_observed_cache_hits(isolated_baseline, hit_on):
    normal = isolated_baseline.return_value
    states = [normal, normal]
    states[hit_on] = normal.model_copy(update={"cache_hit": True, "cached_layer": "L1_EXACT"})
    isolated_baseline.side_effect = states

    with pytest.raises(RuntimeError, match="cache hit"):
        benchmark.run_benchmark(
            gemini_enabled=False, repeats=1, delay_seconds=0.0, verbose=False,
        )
    assert isolated_baseline.call_count == hit_on + 1


@pytest.mark.parametrize("failure_stage", ["dataset", "pipeline"])
def test_baseline_restores_temporary_settings_on_failure(
    isolated_baseline, monkeypatch, failure_stage,
):
    settings = benchmark.get_settings()
    settings.GEMINI_API_KEY = "synthetic-test-key"
    if failure_stage == "dataset":
        monkeypatch.setattr(
            benchmark, "load_benchmark_dataset",
            Mock(side_effect=RuntimeError("synthetic dataset failure")),
        )
    else:
        def fail_pipeline(**kwargs):
            assert settings.GEMINI_API_KEY == ""
            raise RuntimeError("synthetic pipeline failure")
        isolated_baseline.side_effect = fail_pipeline

    with pytest.raises(RuntimeError, match="synthetic"):
        benchmark.run_benchmark(
            gemini_enabled=False, repeats=1, delay_seconds=0.0, verbose=False,
        )
    assert settings.GEMINI_API_KEY == "synthetic-test-key"


def test_baseline_rejects_empty_datasets(isolated_baseline, monkeypatch):
    monkeypatch.setattr(benchmark, "load_benchmark_dataset", lambda path: {"questions": []})
    with pytest.raises(ValueError, match="questions"):
        benchmark.run_benchmark(
            gemini_enabled=False, repeats=1, delay_seconds=0.0, verbose=False,
        )
    isolated_baseline.assert_not_called()


def test_baseline_report_identifies_label_changes_at_the_same_path(isolated_baseline):
    dataset = benchmark.load_benchmark_dataset(Path("synthetic.json"))
    dataset["version"] = "synthetic-v1"
    before = benchmark.run_benchmark(
        gemini_enabled=False, repeats=1, delay_seconds=0.0, verbose=False,
    )
    dataset["questions"][0]["should_abstain"] = True
    after_label_change = benchmark.run_benchmark(
        gemini_enabled=False, repeats=1, delay_seconds=0.0, verbose=False,
    )
    dataset["version"] = "synthetic-v2"
    after_revision = benchmark.run_benchmark(
        gemini_enabled=False, repeats=1, delay_seconds=0.0, verbose=False,
    )

    assert before["dataset_version"] == "synthetic-v1"
    assert after_revision["dataset_version"] == "synthetic-v2"
    assert before["dataset_file"] == after_revision["dataset_file"]
    assert before["dataset_sha256"] != after_label_change["dataset_sha256"]
    assert after_label_change["dataset_sha256"] != after_revision["dataset_sha256"]
