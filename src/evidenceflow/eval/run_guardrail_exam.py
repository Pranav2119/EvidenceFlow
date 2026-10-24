"""
run_guardrail_exam.py: Automated Guardrail Benchmark Exam & Threshold Tuning Runner.
Evaluates the 40-question benchmark dataset against the full EvidenceFlow LangGraph workflow,
measuring:
  - Wrong Refusals (False Positives): Legitimate questions incorrectly refused.
  - Wrong Answers (False Negatives): Adversarial / made-up questions incorrectly answered.
  - Telemetry Code Breakdown: INJECTION, OUT_OF_SCOPE, MISSING_KEYWORD, LOW_SCORE, etc.
  - Empirical Vector Cosine Threshold Tuning across [0.50, 0.55, 0.60, 0.65, 0.70].
"""

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from evidenceflow.config import get_settings
from evidenceflow.graph.workflow import run_pipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_exam_dataset(path: Any) -> List[Dict[str, Any]]:
    """Loads benchmark questions from JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Exam dataset not found at {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", [])


def run_single_evaluation(
    questions: List[Dict[str, Any]],
    threshold: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Executes all questions through run_pipeline() at a given threshold.
    """
    settings = get_settings()
    original_threshold = settings.RETRIEVAL_SCORE_THRESHOLD
    if threshold is not None:
        settings.RETRIEVAL_SCORE_THRESHOLD = threshold

    results = []
    wrong_refusals = []
    wrong_answers = []
    correct_answers = []
    correct_refusals = []
    code_counter: Counter = Counter()

    for idx, item in enumerate(questions, start=1):
        qid = item["id"]
        category = item["category"]
        query = item["query"]
        version = item.get("version", "16")
        expected_behavior = item["expected_behavior"]  # 'answer' or 'abstain'

        try:
            state = run_pipeline(query=query, version=version)
            is_abstained = state.is_abstained
            code = state.abstention_code
            reason = state.abstention_reason
            draft_answer = state.draft_answer
        except Exception as e:
            logger.error(f"Error processing question {qid}: {e}")
            is_abstained = True
            code = "PIPELINE_ERROR"
            reason = str(e)
            draft_answer = ""

        actual_behavior = "abstain" if is_abstained else "answer"
        if code:
            code_counter[code] += 1

        error_type = None
        if expected_behavior == "answer":
            if is_abstained:
                error_type = "WRONG_REFUSAL"
                wrong_refusals.append({
                    "id": qid,
                    "query": query,
                    "version": version,
                    "code": code,
                    "reason": reason,
                })
            else:
                correct_answers.append(qid)
        elif expected_behavior == "abstain":
            if not is_abstained:
                error_type = "WRONG_ANSWER"
                wrong_answers.append({
                    "id": qid,
                    "query": query,
                    "version": version,
                    "answer_preview": draft_answer[:120],
                })
            else:
                correct_refusals.append(qid)

        is_correct = error_type is None
        res_record = {
            "id": qid,
            "category": category,
            "query": query,
            "version": version,
            "expected_behavior": expected_behavior,
            "actual_behavior": actual_behavior,
            "actual_code": code,
            "is_correct": is_correct,
            "error_type": error_type,
            "reason": reason,
        }
        results.append(res_record)

        if verbose:
            status_icon = "PASS" if is_correct else f"FAIL ({error_type})"
            print(f"[{idx:02d}/40] {qid:<5} | {category:<18} | {status_icon:<18} | {query[:45]}...", flush=True)

    # Restore original threshold
    settings.RETRIEVAL_SCORE_THRESHOLD = original_threshold

    total = len(questions)
    total_legitimate = len(correct_answers) + len(wrong_refusals)
    total_adversarial = len(correct_refusals) + len(wrong_answers)

    fp_rate = (len(wrong_refusals) / total_legitimate * 100) if total_legitimate else 0.0
    fn_rate = (len(wrong_answers) / total_adversarial * 100) if total_adversarial else 0.0
    accuracy = ((len(correct_answers) + len(correct_refusals)) / total * 100) if total else 0.0

    summary = {
        "threshold": threshold or original_threshold,
        "total_questions": total,
        "total_legitimate": total_legitimate,
        "total_adversarial": total_adversarial,
        "correct_answers": len(correct_answers),
        "correct_refusals": len(correct_refusals),
        "wrong_refusals_count": len(wrong_refusals),
        "wrong_refusals_rate_pct": round(fp_rate, 2),
        "wrong_answers_count": len(wrong_answers),
        "wrong_answers_rate_pct": round(fn_rate, 2),
        "overall_accuracy_pct": round(accuracy, 2),
        "code_breakdown": dict(code_counter),
        "wrong_refusals": wrong_refusals,
        "wrong_answers": wrong_answers,
        "detailed_results": results,
    }

    return summary


def print_single_report(summary: Dict[str, Any]):
    """Pretty prints the guardrail exam results."""
    print("\n" + "=" * 80)
    print(f" EVIDENCEFLOW GUARDRAIL BENCHMARK EXAM (Threshold: {summary['threshold']:.2f})")
    print("=" * 80)
    print(f"  Total Questions Tested  : {summary['total_questions']}")
    print(f"  Overall Accuracy        : {summary['overall_accuracy_pct']:.1f}%")
    print("-" * 80)
    print("  SECURITY METRICS:")
    print(f"    - Wrong Answers (FN)  : {summary['wrong_answers_count']} / {summary['total_adversarial']} "
          f"({summary['wrong_answers_rate_pct']}%)  <-- DANGEROUS")
    print(f"    - Wrong Refusals (FP) : {summary['wrong_refusals_count']} / {summary['total_legitimate']} "
          f"({summary['wrong_refusals_rate_pct']}%)  <-- Annoying for users")
    print("-" * 80)
    print("  REFUSAL TELEMETRY CODE BREAKDOWN:")
    for code, count in sorted(summary["code_breakdown"].items(), key=lambda x: -x[1]):
        print(f"    {code:<22} : {count:2d}")
    print("-" * 80)

    if summary["wrong_refusals"]:
        print("\n  [!] WRONG REFUSALS (Legitimate questions blocked):")
        for wr in summary["wrong_refusals"]:
            print(f"      [{wr['id']}] (Code: {wr['code']}) Query: \"{wr['query']}\"")
            print(f"          Reason: {wr['reason']}")

    if summary["wrong_answers"]:
        print("\n  [CRITICAL] WRONG ANSWERS (Bad queries slipped through):")
        for wa in summary["wrong_answers"]:
            print(f"      [{wa['id']}] Query: \"{wa['query']}\"")
            print(f"          Answer: {wa['answer_preview']}")

    print("=" * 80 + "\n")


def run_threshold_sweep(
    questions: List[Dict[str, Any]],
    thresholds: List[float],
) -> List[Dict[str, Any]]:
    """
    Evaluates candidate thresholds and displays comparison matrix.
    """
    print("\n" + "=" * 85)
    print(" EXAM-BASED RETRIEVAL THRESHOLD TUNING SWEEP")
    print("=" * 85)
    print(f"Evaluating candidate thresholds: {thresholds}\n")

    sweep_results = []
    for t in thresholds:
        print(f"--> Running 40-question exam for threshold {t:.2f}...", flush=True)
        res = run_single_evaluation(questions, threshold=t, verbose=False)
        sweep_results.append(res)

    print("\n" + "=" * 85)
    print(f"{'Threshold':<11} | {'Wrong Answers (FN)':<20} | {'Wrong Refusals (FP)':<20} | {'Accuracy':<10} | {'Recommendation'}")
    print("-" * 85)

    best_threshold = None
    min_total_errors = float("inf")

    for r in sweep_results:
        t = r["threshold"]
        wa = r["wrong_answers_count"]
        wr = r["wrong_refusals_count"]
        acc = r["overall_accuracy_pct"]
        total_errors = wa * 5 + wr  # Heavy penalty on dangerous wrong answers

        status = ""
        if wa == 0 and wr == 0:
            status = "PERFECT (Optimal)"
        elif wa == 0:
            status = "Viable (Zero Leakage)"
        else:
            status = "UNSAFE (Leakage Detected)"

        if total_errors < min_total_errors:
            min_total_errors = total_errors
            best_threshold = t

        print(f"{t:<11.2f} | {wa:<20d} | {wr:<20d} | {acc:<9.1f}% | {status}")

    print("=" * 85)
    print(f"Selected Optimal Threshold: {best_threshold:.2f}\n")
    return sweep_results


def main():
    parser = argparse.ArgumentParser(description="EvidenceFlow Guardrail Exam & Tuning Runner")
    repo_root = Path(__file__).resolve().parents[3]
    default_exam_path = repo_root / "data" / "benchmarks" / "guardrail_exam_40.json"

    parser.add_argument("--exam-path", type=Path, default=default_exam_path, help="Path to exam JSON")
    parser.add_argument("--tune", action="store_true", help="Run full threshold tuning sweep across [0.50..0.70]")
    parser.add_argument("--threshold", type=float, default=None, help="Specific cosine threshold to evaluate")
    parser.add_argument("--output", type=Path, default=None, help="Output results JSON path")
    parser.add_argument("--verbose", action="store_true", help="Print per-question trace")

    args = parser.parse_args()

    questions = load_exam_dataset(args.exam_path)

    if args.tune:
        candidates = [0.50, 0.55, 0.60, 0.65, 0.70]
        sweep_results = run_threshold_sweep(questions, candidates)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(sweep_results, f, indent=2)
            print(f"Saved sweep results to {args.output}")
    else:
        summary = run_single_evaluation(questions, threshold=args.threshold, verbose=args.verbose)
        print_single_report(summary)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Saved exam summary to {args.output}")


if __name__ == "__main__":
    main()
