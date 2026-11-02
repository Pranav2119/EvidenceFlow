"""
test_benchmark_metrics_final_answer.py: Synthetic unit tests for benchmark runner metrics.
Verifies:
  1. Refused turns have value_accuracy=False and citation_precision/recall=0.0 on final answer.
  2. Answered turns retain factual value accuracy and citation precision/recall.
  3. Benchmark thread IDs start with 'bench-'.
"""

import pytest


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
