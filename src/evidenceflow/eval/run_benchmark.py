"""
run_benchmark.py: Comprehensive Tier 1 & Tier 2 Benchmark Execution Runner for EvidenceFlow AI.
Evaluates version-grounded questions against the baseline pipeline, recording:
  - Guardrail Error Rates: Wrong Refusals (FP) and Wrong Answers (FN)
  - Information Retrieval (IR) Metrics: Recall@5, MRR (computed on answerable queries only)
  - Quality Metrics: Value Accuracy (with unit equivalence) & Citation Precision/Recall
  - Pipeline Self-Check: L1 Groundedness
  - Operational Latency: 3 repeats (warm-up discarded), Median and P95 latency (ms)
Supports running Dev (40 questions) or Frozen Test (20 questions) splits with Gemini ON/OFF
and Memory ON/OFF.
"""

import argparse
from collections import Counter
import json
import logging
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set

from evidenceflow.config import get_settings
from evidenceflow.graph.workflow import run_pipeline
from evidenceflow.eval.metrics import (
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_citation_precision,
    calculate_citation_recall,
    check_value_accuracy,
    is_wrong_refusal,
    is_wrong_answer,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Returns the current short git commit hash."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return output
    except Exception:
        return "unknown"


def load_benchmark_dataset(dataset_path: Path) -> Dict[str, Any]:
    """Loads benchmark JSON file and returns dictionary."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_percentiles(values: List[float], percentiles: List[float]) -> Dict[float, float]:
    """Calculates requested percentiles for a list of numeric values."""
    if not values:
        return {p: 0.0 for p in percentiles}
    sorted_vals = sorted(values)
    results = {}
    for p in percentiles:
        k = (len(sorted_vals) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            results[p] = sorted_vals[int(k)]
        else:
            d0 = sorted_vals[int(f)] * (c - k)
            d1 = sorted_vals[int(c)] * (k - f)
            results[p] = d0 + d1
    return results


def run_benchmark(
    split: str = "dev",
    dataset_file: Optional[str] = None,
    gemini_enabled: bool = True,
    repeats: int = 3,
    delay_seconds: float = 1.5,
    verbose: bool = True,
    enable_memory: bool = False,
) -> Dict[str, Any]:
    """
    Executes the benchmark suite over the designated split or file.
    """
    settings = get_settings()
    original_gemini_key = settings.GEMINI_API_KEY
    if not gemini_enabled:
        settings.GEMINI_API_KEY = ""

    # Select dataset file
    if dataset_file:
        file_path = Path(dataset_file)
    elif split == "test":
        file_path = Path("data/benchmarks/benchmark_test_frozen_20.json")
    else:
        file_path = Path("data/benchmarks/benchmark_dev_40.json")

    data = load_benchmark_dataset(file_path)
    questions = data.get("questions", [])
    total_q = len(questions)

    print("\n" + "=" * 76)
    print(f" EvidenceFlow AI — Evaluation Benchmark ({split.upper()} Split)")
    print("=" * 76)
    print(f"Dataset File:     {file_path.name} ({total_q} questions)")
    print(f"Gemini Mode:      {'ON (Live LLM)' if gemini_enabled else 'OFF (Deterministic Fallback)'}")
    print(f"Memory Mode:      {'ON (Neo4j Conversational Memory)' if enable_memory else 'OFF (Direct Single-Turn)'}")
    print(f"Latency Repeats:  {repeats} runs (warm-up discarded)")
    print(f"Query Pacing:     {delay_seconds}s delay")
    print(f"Score Threshold:  {settings.RETRIEVAL_SCORE_THRESHOLD}")
    print("=" * 76 + "\n")

    question_results = []
    answerable_count = 0
    should_refuse_count = 0

    wrong_refusals = []
    wrong_answers = []
    correct_answers = []
    correct_refusals = []

    recalls: List[float] = []
    mrrs: List[float] = []
    citation_precisions: List[float] = []
    citation_recalls: List[float] = []
    value_accuracies: List[bool] = []
    l1_self_checks: List[bool] = []
    measured_latencies_all: List[float] = []

    for idx, q_data in enumerate(questions, start=1):
        qid = q_data["id"]
        query = q_data["query"]
        version = q_data.get("version")
        should_abstain = q_data.get("should_abstain", False)
        expected_value = q_data.get("expected_value")
        gt_chunks = q_data.get("ground_truth_chunks", [])
        gt_matcher = q_data.get("ground_truth_passage_matcher")
        category = q_data.get("category", "unknown")

        if should_abstain:
            should_refuse_count += 1
        else:
            answerable_count += 1

        # -----------------------------------------------------------------
        # Latency Repeats: 1 Warm-up (discarded) + 'repeats' timed executions
        # -----------------------------------------------------------------
        # Warm-up run
        final_state = run_pipeline(
            query=query,
            version=version,
            thread_id=f"bench_{split}_{qid}_warmup",
            use_postgres_checkpointer=False,
            enable_memory=enable_memory,
        )

        timed_latencies = []
        for r in range(repeats):
            if gemini_enabled and delay_seconds > 0:
                time.sleep(delay_seconds)

            t_start = time.perf_counter()
            exec_state = run_pipeline(
                query=query,
                version=version,
                thread_id=f"bench_{split}_{qid}_r{r}",
                use_postgres_checkpointer=False,
                enable_memory=enable_memory,
            )
            t_end = time.perf_counter()
            duration_ms = (t_end - t_start) * 1000.0
            timed_latencies.append(duration_ms)
            final_state = exec_state

        measured_latencies_all.extend(timed_latencies)
        median_lat = calculate_percentiles(timed_latencies, [50])[50]

        # Extract pipeline state variables
        is_abstained = final_state.is_abstained
        abstention_code = final_state.abstention_code
        abstention_reason = final_state.abstention_reason
        retrieved_passages = final_state.retrieved_passages
        draft_answer = final_state.draft_answer
        citations = final_state.citations
        l1_passed = final_state.l1_passed

        retrieved_ids: Set[str] = {
            p.get("chunk_id") or p.get("id")
            for p in retrieved_passages
            if (p.get("chunk_id") or p.get("id"))
        }

        # -----------------------------------------------------------------
        # Error & Metric Evaluation
        # -----------------------------------------------------------------
        error_class = None
        q_recall: Optional[float] = None
        q_mrr: Optional[float] = None
        q_prec: Optional[float] = None
        q_crecall: Optional[float] = None
        q_val_acc: Optional[bool] = None

        if should_abstain:
            # Question was expected to be refused
            if is_abstained:
                correct_refusals.append(qid)
            else:
                error_class = "WRONG_ANSWER"
                wrong_answers.append(qid)
        else:
            # Question was answerable
            if is_abstained:
                error_class = "WRONG_REFUSAL"
                wrong_refusals.append(qid)
            else:
                correct_answers.append(qid)

            # IR and Answer Quality metrics apply to answerable questions
            q_recall = calculate_recall_at_k(retrieved_passages, gt_chunks, gt_matcher, k=5)
            q_mrr = calculate_reciprocal_rank(retrieved_passages, gt_chunks, gt_matcher, k=5)
            q_prec = calculate_citation_precision(citations, retrieved_ids)
            q_crecall = calculate_citation_recall(draft_answer, citations)
            q_val_acc = check_value_accuracy(draft_answer, expected_value)

            recalls.append(q_recall)
            mrrs.append(q_mrr)
            citation_precisions.append(q_prec)
            citation_recalls.append(q_crecall)
            value_accuracies.append(q_val_acc)
            l1_self_checks.append(l1_passed)

        status_str = "PASS" if error_class is None else f"FAIL ({error_class})"
        if verbose:
            short_q = (query[:38] + "..") if len(query) > 40 else query
            val_acc_str = f"Val: {'Y' if q_val_acc else 'N'}" if q_val_acc is not None else "Val: N/A"
            r5_str = f"R@5: {q_recall:.2f}" if q_recall is not None else "R@5: N/A"
            print(f"[{idx:02d}/{total_q}] {qid:<4} | {category:<19} | {status_str:<19} | {median_lat:6.1f}ms | {r5_str:<9} | {val_acc_str:<8} | {short_q}")

        question_results.append({
            "id": qid,
            "category": category,
            "query": query,
            "version": version,
            "should_abstain": should_abstain,
            "is_abstained": is_abstained,
            "abstention_code": abstention_code,
            "abstention_reason": abstention_reason,
            "error_class": error_class,
            "latencies_ms": timed_latencies,
            "median_latency_ms": median_lat,
            "recall_at_5": q_recall,
            "mrr": q_mrr,
            "citation_precision": q_prec,
            "citation_recall": q_crecall,
            "value_accuracy": q_val_acc,
            "l1_self_check": l1_passed if not should_abstain else None,
            "answer_preview": draft_answer[:160],
        })

    # Restore settings
    settings.GEMINI_API_KEY = original_gemini_key

    # -----------------------------------------------------------------
    # Aggregate Metrics Calculation
    # -----------------------------------------------------------------
    overall_correct = len(correct_answers) + len(correct_refusals)
    overall_acc_pct = (overall_correct / total_q) * 100.0 if total_q else 0.0

    wrong_refusal_count = len(wrong_refusals)
    wrong_refusal_pct = (wrong_refusal_count / answerable_count) * 100.0 if answerable_count else 0.0

    wrong_answer_count = len(wrong_answers)
    wrong_answer_pct = (wrong_answer_count / should_refuse_count) * 100.0 if should_refuse_count else 0.0

    mean_recall = (sum(recalls) / len(recalls)) * 100.0 if recalls else 0.0
    mean_mrr = (sum(mrrs) / len(mrrs)) if mrrs else 0.0
    mean_precision = (sum(citation_precisions) / len(citation_precisions)) * 100.0 if citation_precisions else 0.0
    mean_cit_recall = (sum(citation_recalls) / len(citation_recalls)) * 100.0 if citation_recalls else 0.0

    val_acc_count = sum(1 for v in value_accuracies if v)
    val_acc_pct = (val_acc_count / len(value_accuracies)) * 100.0 if value_accuracies else 0.0

    l1_count = sum(1 for v in l1_self_checks if v)
    l1_pct = (l1_count / len(l1_self_checks)) * 100.0 if l1_self_checks else 0.0

    lat_percentiles = calculate_percentiles(measured_latencies_all, [50, 95])

    summary = {
        "split": split,
        "dataset_file": file_path.name,
        "git_commit": get_git_commit_hash(),
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.GEMINI_EMBEDDING_MODEL if settings.EMBEDDING_PROVIDER == "gemini" else "BAAI/bge-small-en-v1.5",
        "vector_score_threshold": settings.RETRIEVAL_SCORE_THRESHOLD,
        "gemini_mode": "ON" if gemini_enabled else "OFF (Deterministic Fallback)",
        "memory_mode": "ON (Neo4j)" if enable_memory else "OFF",
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "repeats": repeats,
        "delay_seconds": delay_seconds,
        "counts": {
            "total_questions": total_q,
            "answerable_questions": answerable_count,
            "should_refuse_questions": should_refuse_count,
            "correct_answers": len(correct_answers),
            "correct_refusals": len(correct_refusals),
            "wrong_refusals": wrong_refusal_count,
            "wrong_answers": wrong_answer_count,
            "value_accurate": val_acc_count,
            "l1_self_check_passed": l1_count,
        },
        "metrics": {
            "overall_accuracy_pct": round(overall_acc_pct, 2),
            "wrong_refusal_pct": round(wrong_refusal_pct, 2),
            "wrong_answer_pct": round(wrong_answer_pct, 2),
            "mean_recall_at_5_pct": round(mean_recall, 2),
            "mean_mrr": round(mean_mrr, 4),
            "citation_precision_pct": round(mean_precision, 2),
            "citation_recall_pct": round(mean_cit_recall, 2),
            "value_accuracy_pct": round(val_acc_pct, 2),
            "l1_self_check_pct": round(l1_pct, 2),
            "latency_median_ms": round(lat_percentiles[50], 1),
            "latency_p95_ms": round(lat_percentiles[95], 1),
        },
        "question_results": question_results,
    }

    # Print Formatted Markdown Table
    print("\n" + "=" * 76)
    print(" EvidenceFlow AI — Evaluation Benchmark Final Results Table")
    print("=" * 76)
    print(f"| Metric | Count | Percentage / Value | Target / Classification |")
    print(f"| :--- | :---: | :---: | :--- |")
    print(f"| **Overall Accuracy** | {overall_correct} / {total_q} | **{overall_acc_pct:.1f}%** | Overall End-to-End |")
    print(f"| **Wrong Refusals (FP, $\\alpha$-error)** | {wrong_refusal_count} / {answerable_count} | **{wrong_refusal_pct:.1f}%** | Annoying user errors |")
    print(f"| **Wrong Answers (FN, $\\beta$-error)** | {wrong_answer_count} / {should_refuse_count} | **{wrong_answer_pct:.1f}%** | Dangerous leakages |")
    print(f"| **Retrieval Recall@5** | — | **{mean_recall:.1f}%** | Top-5 Ground Truth Coverage |")
    print(f"| **Retrieval MRR** | — | **{mean_mrr:.4f}** | Mean Reciprocal Rank |")
    print(f"| **Value Accuracy** | {val_acc_count} / {answerable_count} | **{val_acc_pct:.1f}%** | Unit Equivalence Verified |")
    print(f"| **Citation Precision** | — | **{mean_precision:.1f}%** | Verified Quoted Footnotes |")
    print(f"| **Citation Recall** | — | **{mean_cit_recall:.1f}%** | Claims with Citations |")
    print(f"| **L1 Groundedness (Self-Check)** | {l1_count} / {answerable_count} | **{l1_pct:.1f}%** | In-loop Pipeline Verifier |")
    print(f"| **Median Latency** | — | **{lat_percentiles[50]:.1f} ms** | Across {repeats} repeats |")
    print(f"| **P95 Latency** | — | **{lat_percentiles[95]:.1f} ms** | Tail latency |")
    print("=" * 76 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Evaluation Benchmark.")
    parser.add_argument("--split", choices=["dev", "test"], default="dev", help="Dataset split (dev or test)")
    parser.add_argument("--dataset", type=str, default=None, help="Explicit dataset JSON file path")
    parser.add_argument("--tier", type=int, default=1, help="Evaluation tier")
    parser.add_argument("--repeats", type=int, default=3, help="Latency repeat runs (warm-up discarded)")
    parser.add_argument("--delay", type=float, default=1.5, help="Query pacing delay in seconds")
    parser.add_argument("--no-gemini", action="store_true", help="Force offline mode with Gemini disabled")
    parser.add_argument("--enable-memory", action="store_true", help="Enable Neo4j conversational memory")
    parser.add_argument("--output", type=str, default=None, help="Output JSON results path")
    args = parser.parse_args()

    results = run_benchmark(
        split=args.split,
        dataset_file=args.dataset,
        gemini_enabled=not args.no_gemini,
        repeats=args.repeats,
        delay_seconds=args.delay,
        verbose=True,
        enable_memory=args.enable_memory,
    )

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to: {out_p}")


if __name__ == "__main__":
    main()
