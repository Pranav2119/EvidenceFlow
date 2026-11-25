"""
test_eval_metrics.py: Unit tests and failure edge-case tests for evaluation metrics.
Verifies zero-division protection, dual ground-truth matching, unit-equivalence value accuracy,
and error classification.
"""

import hashlib
import json
from pathlib import Path

import pytest
from evidenceflow.eval.metrics import (
    matches_ground_truth_passage,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_citation_precision,
    calculate_citation_recall,
    check_value_accuracy,
    is_wrong_refusal,
    is_wrong_answer,
)
from evidenceflow.state import Citation


def test_zero_division_guard_empty_citations():
    """Verify that empty citations return 0.0 without ZeroDivisionError."""
    precision = calculate_citation_precision(citations=[], retrieved_chunk_ids={"chunk-01"})
    assert precision == 0.0

    recall = calculate_citation_recall(draft_answer="", citations=[])
    assert recall == 0.0

    recall_with_answer_no_citations = calculate_citation_recall(
        draft_answer="The default is 10000.",
        citations=[]
    )
    assert recall_with_answer_no_citations == 0.0


def test_zero_division_guard_empty_ground_truth():
    """Verify that empty ground truth returns 0.0 without ZeroDivisionError."""
    retrieved = [
        {"chunk_id": "chunk-01", "version": "16", "doc_name": "resource.html", "title": "vacuum_cost_limit"}
    ]
    recall = calculate_recall_at_k(retrieved_chunks=retrieved, gt_chunks=[], gt_matcher=None)
    assert recall == 0.0

    mrr = calculate_reciprocal_rank(retrieved_chunks=retrieved, gt_chunks=[], gt_matcher=None)
    assert mrr == 0.0


def test_matches_ground_truth_passage_dual():
    """Verify both exact chunk_id match and composite tuple fallback match."""
    chunk = {
        "chunk_id": "pg16-release-16-c86f660e37",
        "version": "16",
        "doc_name": "release-16.html",
        "section": "Release 16",
        "title": "Remove the server variable vacuum_defer_cleanup_age",
        "content": "Remove the server variable vacuum_defer_cleanup_age (Andres Freund)"
    }

    # 1. Direct chunk ID match
    assert matches_ground_truth_passage(chunk, gt_chunks=["pg16-release-16-c86f660e37"])

    # 2. Composite tuple match (even if hash differs!)
    matcher = {
        "version": "16",
        "doc_name": "release-16.html",
        "title_contains": "vacuum_defer_cleanup_age"
    }
    assert matches_ground_truth_passage(chunk, gt_chunks=["pg16-release-16-96b6fa3f68"], gt_matcher=matcher)

    # 3. Mismatched version
    mismatched_matcher = {
        "version": "14",
        "doc_name": "release-16.html",
        "title_contains": "vacuum_defer_cleanup_age"
    }
    assert not matches_ground_truth_passage(chunk, gt_chunks=[], gt_matcher=mismatched_matcher)


def test_calculate_recall_and_mrr():
    """Verify Recall@5 and reciprocal rank calculations."""
    retrieved = [
        {"chunk_id": "irrelevant-01"},
        {"chunk_id": "pg16-runtime-config-resource-vacuum-cost-limit"},
        {"chunk_id": "irrelevant-02"},
    ]
    gt_chunks = ["pg16-runtime-config-resource-vacuum-cost-limit"]

    # Found at rank 2
    recall = calculate_recall_at_k(retrieved, gt_chunks, k=5)
    mrr = calculate_reciprocal_rank(retrieved, gt_chunks, k=5)

    assert recall == 1.0
    assert mrr == 0.5  # 1 / 2

    # If k=1, not found in top-1
    recall_top1 = calculate_recall_at_k(retrieved, gt_chunks, k=1)
    mrr_top1 = calculate_reciprocal_rank(retrieved, gt_chunks, k=1)
    assert recall_top1 == 0.0
    assert mrr_top1 == 0.0


def test_citation_precision():
    """Verify Citation Precision calculation."""
    retrieved_ids = {"chunk-01", "chunk-02"}
    citations = [
        Citation(chunk_id="chunk-01", doc_name="doc.html", section="sec", quote="q", is_verified=True),
        Citation(chunk_id="chunk-02", doc_name="doc.html", section="sec", quote="q", is_verified=False),  # unverified
        Citation(chunk_id="chunk-03", doc_name="doc.html", section="sec", quote="q", is_verified=True),  # not in retrieved
    ]

    precision = calculate_citation_precision(citations, retrieved_ids)
    # Only chunk-01 is verified AND in retrieved context -> 1 / 3 = 0.3333
    assert abs(precision - (1.0 / 3.0)) < 1e-4


def test_value_accuracy_unit_equivalence():
    """Verify expected value matching with unit equivalence."""
    # Time unit equivalence
    assert check_value_accuracy("The timeout is set to 60s by default.", expected_value="1min")
    assert check_value_accuracy("Autovacuum naptime is 1 min.", expected_value="60s")

    # Memory unit equivalence
    assert check_value_accuracy("Allocated 1 MB for each session.", expected_value="1024kB")
    assert check_value_accuracy("Default size is 128MB.", expected_value="128MB")

    # Raw numbers & booleans
    assert check_value_accuracy("The vacuum_cost_limit is 10000 in PG16.", expected_value="10000")
    assert check_value_accuracy("By default, log_checkpoints is on.", expected_value="on")
    assert not check_value_accuracy("By default, log_checkpoints is off.", expected_value="on")

    # None expected value (unspecified) returns True
    assert check_value_accuracy("Any answer.", expected_value=None)


@pytest.mark.parametrize(
    "answer, expected_value",
    [
        ("shared_buffers defaults to 128MB.", "8MB"),
        ("The connection limit is unchanged.", "on"),
        ("The default is -1.", "1"),
        ("The default is 0.1.", "1"),
        ("The default is 1000.", "100"),
        ("See the documentation [^pg16-setting-100].", "100"),
    ],
)
def test_value_accuracy_rejects_partial_values_and_citation_ids(answer, expected_value):
    assert check_value_accuracy(answer, expected_value) is False


@pytest.mark.parametrize(
    "answer, expected_value",
    [
        ("The default is -1.", "-1"),
        ("The default is 100.", "100"),
        ("The allocation is 1 MB.", "1024kB"),
        ("The threshold is 0.1.", "0.1"),
    ],
)
def test_value_accuracy_keeps_complete_value_matches(answer, expected_value):
    assert check_value_accuracy(answer, expected_value) is True


def test_error_classification():
    """Verify Wrong Refusal (FP) and Wrong Answer (FN) boolean classifications."""
    # Legitimate question (should_abstain=False)
    assert not is_wrong_refusal(should_abstain=False, is_abstained=False)
    assert is_wrong_refusal(should_abstain=False, is_abstained=True)  # Wrong Refusal!
    assert not is_wrong_answer(should_abstain=False, is_abstained=False)

    # Malicious/unanswerable question (should_abstain=True)
    assert not is_wrong_answer(should_abstain=True, is_abstained=True)
    assert is_wrong_answer(should_abstain=True, is_abstained=False)  # Wrong Answer!
    assert not is_wrong_refusal(should_abstain=True, is_abstained=False)


def test_dataset_fingerprint_tracks_labels_not_object_key_order():
    from evidenceflow.eval.metrics import get_dataset_metadata

    original = {"version": "1.1", "questions": [{"id": "q1", "expected_value": "1"}]}
    reordered = {"questions": [{"expected_value": "1", "id": "q1"}], "version": "1.1"}
    changed = {"version": "1.1", "questions": [{"id": "q1", "expected_value": "-1"}]}

    assert get_dataset_metadata(original) == get_dataset_metadata(reordered)
    assert get_dataset_metadata(original)["dataset_sha256"] != get_dataset_metadata(changed)["dataset_sha256"]


def test_development_answerable_labels_have_exact_same_version_evidence():
    root = Path(__file__).resolve().parents[1]
    dataset = json.loads((root / "data" / "benchmarks" / "benchmark_dev_40.json").read_text(encoding="utf-8"))
    with (root / "data" / "processed" / "chunks.jsonl").open(encoding="utf-8") as source:
        corpus = {passage["id"]: passage for passage in map(json.loads, source)}

    for question in dataset["questions"]:
        if question["should_abstain"]:
            assert question["expected_value"] is None, question["id"]
            assert question["ground_truth_chunks"] == [], question["id"]
            continue

        quote = question.get("ground_truth_quote")
        assert quote and quote.strip(), f"{question['id']} lacks a reviewed support quote"
        passages = [corpus[chunk_id] for chunk_id in question["ground_truth_chunks"]]
        assert passages, question["id"]
        assert all(passage["version"] == question["version"] for passage in passages), question["id"]
        assert any(quote in passage["content"] for passage in passages), question["id"]
        assert check_value_accuracy(quote, question["expected_value"]), question["id"]
        for passage in passages:
            assert hashlib.sha256(passage["content"].encode("utf-8")).hexdigest() == passage["sha256"], question["id"]
