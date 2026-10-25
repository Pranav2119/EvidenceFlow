"""
run_holdout_exam.py: Single-pass, zero-tuning evaluation of the 25-question held-out benchmark.
Evaluates EvidenceFlow on held-out questions with frozen code.
"""

from collections import Counter
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

from evidenceflow.config import get_settings
from evidenceflow.graph.workflow import run_pipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_holdout_dataset(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", [])


def run_holdout_evaluation(exam_path: Path, verbose: bool = True) -> Dict[str, Any]:
    questions = load_holdout_dataset(exam_path)
    settings = get_settings()

    print("\n" + "=" * 90)
    print(" HELD-OUT GUARDRAIL EXAM EVALUATION (25 QUESTIONS - ZERO CODE CHANGES)")
    print("=" * 90)
    print(f"Dataset: {exam_path.name} | Total Questions: {len(questions)}")
    print(f"Retrieval Cosine Threshold: {settings.RETRIEVAL_SCORE_THRESHOLD}")
    print("=" * 90 + "\n")

    results = []
    wrong_refusals = []
    wrong_answers = []
    correct_answers = []
    correct_refusals = []
    either_cases = []
    code_counter: Counter = Counter()

    for idx, item in enumerate(questions, start=1):
        qid = item["id"]
        category = item["category"]
        query = item["question"]
        version = str(item.get("version", "16"))
        expected = item.get("expected")  # 'answer', 'abstain', or 'either'
        expected_values = item.get("expected_values", [])
        expected_param = item.get("expected_param", "")

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

        if expected == "either":
            either_cases.append({
                "id": qid,
                "query": query,
                "behavior": actual_behavior,
                "code": code,
                "reason": reason,
                "answer_preview": draft_answer[:120],
            })
            is_correct = True
        elif expected == "answer":
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
            is_correct = (error_type is None)
        elif expected == "abstain":
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
            is_correct = (error_type is None)
        else:
            is_correct = True

        status_tag = "PASS" if is_correct else f"FAIL ({error_type})"
        if expected == "either":
            status_tag = f"EITHER -> {actual_behavior.upper()}"

        code_str = f" [{code}]" if code else ""
        if verbose:
            print(f"[{idx:02d}/25] {qid} ({category:<18}) -> {actual_behavior.upper()}{code_str:<18} | {status_tag}")
            if error_type:
                print(f"       Query: '{query}' (Postgres {version})")
                if is_abstained:
                    print(f"       Refusal: {reason}")
                else:
                    print(f"       Answer: {draft_answer[:100]}...")

        results.append({
            "id": qid,
            "category": category,
            "query": query,
            "version": version,
            "expected": expected,
            "actual_behavior": actual_behavior,
            "actual_code": code,
            "is_correct": is_correct,
            "error_type": error_type,
            "reason": reason,
            "draft_answer": draft_answer,
            "expected_values": expected_values,
            "expected_param": expected_param,
        })

    # Summary
    scorable_count = len(questions) - len(either_cases)
    wa_count = len(wrong_answers)
    wr_count = len(wrong_refusals)
    correct_count = len(correct_answers) + len(correct_refusals)
    acc_pct = (correct_count / scorable_count * 100.0) if scorable_count > 0 else 0.0

    print("\n" + "=" * 90)
    print(" HELD-OUT EXAM RESULTS SUMMARY")
    print("=" * 90)
    print(f"Total Questions Evaluated: {len(questions)}")
    print(f"Scorable Questions:        {scorable_count} (Excludes {len(either_cases)} 'either' question)")
    print(f"Correct Answers:           {len(correct_answers)}")
    print(f"Correct Refusals:          {len(correct_refusals)}")
    print(f"Wrong Refusals (FP):       {wr_count} ({(wr_count / scorable_count * 100):.1f}%)")
    print(f"Wrong Answers (FN):        {wa_count} ({(wa_count / scorable_count * 100):.1f}%)")
    print(f"Overall Accuracy:          {acc_pct:.1f}%")
    print("\nRefusal Code Telemetry Breakdown:")
    for c, cnt in sorted(code_counter.items()):
        print(f"  - {c:<22}: {cnt} fires")
    print("=" * 90 + "\n")

    return {
        "total_questions": len(questions),
        "scorable_count": scorable_count,
        "overall_accuracy_pct": round(acc_pct, 2),
        "wrong_answers_count": wa_count,
        "wrong_refusals_count": wr_count,
        "correct_answers_count": len(correct_answers),
        "correct_refusals_count": len(correct_refusals),
        "code_breakdown": dict(code_counter),
        "either_cases": either_cases,
        "wrong_answers": wrong_answers,
        "wrong_refusals": wrong_refusals,
        "results": results,
    }


def main():
    repo_root = Path(__file__).resolve().parents[3]
    exam_path = repo_root / "data" / "benchmarks" / "guardrail_holdout_25.json"
    output_path = repo_root / "data" / "benchmarks" / "guardrail_holdout_results.json"

    if not exam_path.exists():
        print(f"Holdout exam file not found at {exam_path}")
        sys.exit(1)

    summary = run_holdout_evaluation(exam_path, verbose=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Holdout evaluation results saved to {output_path}")


if __name__ == "__main__":
    main()
