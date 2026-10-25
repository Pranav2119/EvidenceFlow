"""
run_threshold_spread.py: Multi-run stability and variance analysis across candidate thresholds.
Runs each vector retrieval threshold 3 times to measure variance and report the spread.
"""

from collections import Counter
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, List

from evidenceflow.eval.run_guardrail_exam import load_exam_dataset, run_single_evaluation

logging.basicConfig(level=logging.WARNING)


def calculate_stats(values: List[float]) -> Dict[str, float]:
    """Calculates min, max, mean, and sample standard deviation."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(mean, 2),
        "std": round(std, 2),
    }


def main():
    repo_root = Path(__file__).resolve().parents[3]
    exam_path = repo_root / "data" / "benchmarks" / "guardrail_exam_40.json"
    output_path = repo_root / "data" / "benchmarks" / "guardrail_threshold_spread_3x.json"

    questions = load_exam_dataset(exam_path)
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    num_runs = 3

    print("\n" + "=" * 95, flush=True)
    print(" MULTI-RUN RETRIEVAL THRESHOLD SPREAD & VARIANCE BENCHMARK (3x Repeated)", flush=True)
    print("=" * 95, flush=True)
    print(f"Testing thresholds: {thresholds} | Repeats per threshold: {num_runs}", flush=True)
    print(f"Total pipeline executions: {len(thresholds) * num_runs * len(questions)} queries\n", flush=True)

    all_spread_results = []

    for t in thresholds:
        print(f"--> Evaluating threshold {t:.2f} across {num_runs} runs...", flush=True)
        runs_data = []
        for r_idx in range(1, num_runs + 1):
            sys.stdout.write(f"    [Run {r_idx}/{num_runs}] Running 40 queries... ")
            sys.stdout.flush()
            res = run_single_evaluation(questions, threshold=t, verbose=False)
            runs_data.append(res)
            print(f"Accuracy: {res['overall_accuracy_pct']:.1f}% | WA (FN): {res['wrong_answers_count']} | WR (FP): {res['wrong_refusals_count']}", flush=True)

        accuracies = [r["overall_accuracy_pct"] for r in runs_data]
        wrong_answers = [r["wrong_answers_count"] for r in runs_data]
        wrong_refusals = [r["wrong_refusals_count"] for r in runs_data]

        acc_stats = calculate_stats(accuracies)
        wa_stats = calculate_stats([float(x) for x in wrong_answers])
        wr_stats = calculate_stats([float(x) for x in wrong_refusals])

        spread_record = {
            "threshold": t,
            "num_runs": num_runs,
            "accuracies": accuracies,
            "wrong_answers": wrong_answers,
            "wrong_refusals": wrong_refusals,
            "accuracy_stats": acc_stats,
            "wrong_answers_stats": wa_stats,
            "wrong_refusals_stats": wr_stats,
            "runs": runs_data,
        }
        all_spread_results.append(spread_record)

    print("\n" + "=" * 95, flush=True)
    print(f"{'Threshold':<10} | {'Run 1 (WA/WR)':<15} | {'Run 2 (WA/WR)':<15} | {'Run 3 (WA/WR)':<15} | {'Accuracy Spread':<17} | {'Mean ± Std'}", flush=True)
    print("-" * 95, flush=True)

    for s in all_spread_results:
        t = s["threshold"]
        r1 = f"{s['wrong_answers'][0]} / {s['wrong_refusals'][0]}"
        r2 = f"{s['wrong_answers'][1]} / {s['wrong_refusals'][1]}"
        r3 = f"{s['wrong_answers'][2]} / {s['wrong_refusals'][2]}"
        acc_s = f"{s['accuracy_stats']['min']:.1f}% - {s['accuracy_stats']['max']:.1f}%"
        mean_std = f"{s['accuracy_stats']['mean']:.1f}% ± {s['accuracy_stats']['std']:.2f}%"
        print(f"{t:<10.2f} | {r1:<15} | {r2:<15} | {r3:<15} | {acc_s:<17} | {mean_std}", flush=True)

    print("=" * 95 + "\n", flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_spread_results, f, indent=2)
    print(f"Spread results successfully saved to {output_path}\n", flush=True)


if __name__ == "__main__":
    main()
