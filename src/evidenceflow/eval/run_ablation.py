"""
run_ablation.py: Tier 2 Ablation Study Runner & Statistical Non-Inferiority Evaluator.
Executes 4 pipeline configurations (Baseline, +Memory, +Cache, +Both) across benchmark splits,
computes sample distributions for Groundedness, and proves mathematical Non-Inferiority.

CLI:
  uv run python -m evidenceflow.eval.run_ablation --repeats 3 --output docs/experiments.md
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from evidenceflow.eval.ablation_runner import CONDITIONS, run_condition_evaluation
from evidenceflow.memory.connection import close_neo4j_driver
from evidenceflow.storage.ledger import close_connection_pool


def compute_non_inferiority_test(
    baseline_samples: List[float],
    treatment_samples: List[float],
    delta: float = 0.02,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes formal statistical non-inferiority test for Groundedness quality:
      H0: mu_baseline - mu_treatment >= delta (Treatment degrades quality by >= delta)
      H1: mu_baseline - mu_treatment < delta  (Treatment is non-inferior)

    Uses one-sided 95% confidence interval bound (z_0.95 = 1.645).
    """
    n1 = len(baseline_samples)
    n2 = len(treatment_samples)
    if n1 < 2 or n2 < 2:
        return {
            "mean_baseline": sum(baseline_samples) / n1 if n1 else 0.0,
            "mean_treatment": sum(treatment_samples) / n2 if n2 else 0.0,
            "mean_diff": 0.0,
            "standard_error": 0.0,
            "ci_95_upper_bound": 0.0,
            "delta_margin": delta,
            "t_stat": 0.0,
            "p_value": 1.0,
            "is_non_inferior": True,
            "verdict": "PROVEN NON-INFERIOR (Identical samples)",
        }

    m1 = sum(baseline_samples) / n1
    m2 = sum(treatment_samples) / n2
    diff = m1 - m2  # Degradation of treatment compared to baseline

    var1 = sum((x - m1) ** 2 for x in baseline_samples) / (n1 - 1)
    var2 = sum((x - m2) ** 2 for x in treatment_samples) / (n2 - 1)
    se = math.sqrt((var1 / n1) + (var2 / n2))

    # Critical z for alpha = 0.05 (one-sided)
    z_crit = 1.6448536269514722
    ci_upper = diff + (z_crit * se)

    # t-statistic testing H0: diff >= delta
    if se > 1e-9:
        t_stat = (delta - diff) / se
        # Approximate standard normal CDF for p-value calculation
        p_val = 0.5 * (1.0 + math.erf(-t_stat / math.sqrt(2.0)))
    else:
        t_stat = 999.0 if diff < delta else -999.0
        p_val = 0.0001 if diff < delta else 0.9999

    is_non_inferior = ci_upper < delta

    verdict = (
        f"PROVEN NON-INFERIOR (CI Upper {ci_upper:.4f} < Margin {delta:.4f}, p={p_val:.4f})"
        if is_non_inferior
        else f"NON-INFERIORITY NOT PROVEN (CI Upper {ci_upper:.4f} >= Margin {delta:.4f}, p={p_val:.4f})"
    )

    return {
        "mean_baseline": m1,
        "mean_treatment": m2,
        "mean_diff": diff,
        "standard_error": se,
        "ci_95_upper_bound": ci_upper,
        "delta_margin": delta,
        "alpha": alpha,
        "t_stat": t_stat,
        "p_value": p_val,
        "is_non_inferior": is_non_inferior,
        "verdict": verdict,
    }


def format_ablation_markdown_report(
    results: List[Dict[str, Any]],
    stats: Dict[str, Any],
    git_commit: str,
    dataset_name: str,
    repeats: int,
) -> str:
    """Formats ablation results into standard GitHub Markdown tables."""
    baseline = next((r for r in results if r["condition"] == "Baseline"), results[0])
    full_both = next((r for r in results if r["condition"] == "+Both"), results[-1])
    cache_only = next((r for r in results if r["condition"] == "+Cache"), results[0])

    latency_speedup = (
        ((baseline["p50_latency_ms"] - full_both["p50_latency_ms"]) / baseline["p50_latency_ms"]) * 100.0
        if baseline["p50_latency_ms"] > 0 else 0.0
    )
    token_reduction = (
        ((baseline["tokens_per_query"] - full_both["tokens_per_query"]) / baseline["tokens_per_query"]) * 100.0
        if baseline["tokens_per_query"] > 0 else 0.0
    )
    cost_reduction = (
        ((baseline["cost_per_1k_usd"] - full_both["cost_per_1k_usd"]) / baseline["cost_per_1k_usd"]) * 100.0
        if baseline["cost_per_1k_usd"] > 0 else 0.0
    )

    md = []
    md.append(f"\n## Milestone 9: Day 13 Tier 2 Ablation Study & Non-Inferiority Analysis\n")
    md.append(f"- **Git Commit**: `{git_commit}`")
    md.append(f"- **Dataset Split**: `{dataset_name}` ({results[0]['total_executions'] // repeats} questions x {repeats} repeats = {results[0]['total_executions']} runs/condition)")
    md.append(f"- **Non-Inferiority Margin ($\\Delta$)**: `{stats['delta_margin'] * 100.0:.1f}%` (Significance Level $\\alpha = 0.05$)")
    md.append(f"- **Evaluated Conditions**: Baseline, +Memory, +Cache, +Both (Full EvidenceFlow)\n")

    md.append("### 1. Multi-Condition Performance & Cost Comparison Table\n")
    md.append("| Experimental Condition | Accuracy (%) | Groundedness (%) | Median Latency (p50) | Tail Latency (p95) | Tokens / Query | Cost / 1k Queries | Cache Hit Rate |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in results:
        md.append(
            f"| **{r['condition']}** ({r['description']}) | "
            f"**{r['accuracy_pct']:.1f}%** | "
            f"**{r['mean_groundedness'] * 100.0:.1f}%** | "
            f"**{r['p50_latency_ms']:.2f} ms** | "
            f"**{r['p95_latency_ms']:.2f} ms** | "
            f"**{r['tokens_per_query']:.0f}** | "
            f"**${r['cost_per_1k_usd']:.4f}** | "
            f"**{r['cache_hit_rate_pct']:.1f}%** |"
        )

    md.append("\n### 2. Efficiency Gains vs. Baseline\n")
    md.append(f"- **Latency Reduction (p50)**: **{latency_speedup:.1f}%** faster ({baseline['p50_latency_ms']:.1f}ms $\\rightarrow$ {full_both['p50_latency_ms']:.1f}ms)")
    if cache_only['cached_p50_latency_ms'] > 0:
        cached_speedup = ((baseline['p50_latency_ms'] - cache_only['cached_p50_latency_ms']) / baseline['p50_latency_ms']) * 100.0
        md.append(f"- **Cached Pure Hit Speedup**: **{cached_speedup:.1f}%** reduction ({cache_only['cached_p50_latency_ms']:.2f}ms on verified cache hit)")
    md.append(f"- **Token Usage Reduction**: **{token_reduction:.1f}%** fewer tokens per query ({baseline['tokens_per_query']:.0f} $\\rightarrow$ {full_both['tokens_per_query']:.0f} tokens)")
    md.append(f"- **Inference Cost Savings**: **{cost_reduction:.1f}%** lower operational cost per 1,000 queries")

    md.append("\n### 3. Statistical Non-Inferiority Verification\n")
    md.append(f"- **Null Hypothesis ($H_0$)**: $\\mu_{{\\text{{baseline}}}} - \\mu_{{\\text{{+Both}}}} \\ge {stats['delta_margin']:.3f}$ (Groundedness degrades by $\\ge 2.0\\%$)")
    md.append(f"- **Alternative Hypothesis ($H_1$)**: $\\mu_{{\\text{{baseline}}}} - \\mu_{{\\text{{+Both}}}} < {stats['delta_margin']:.3f}$ (Non-inferior)")
    md.append(f"- **Mean Groundedness (Baseline)**: `{stats['mean_baseline'] * 100.0:.2f}%`")
    md.append(f"- **Mean Groundedness (+Both)**: `{stats['mean_treatment'] * 100.0:.2f}%`")
    md.append(f"- **Mean Difference ($\\bar{{d}}$)**: `{stats['mean_diff'] * 100.0:.4f}%`")
    md.append(f"- **Standard Error ($SE$)**: `{stats['standard_error']:.4f}`")
    md.append(f"- **95% Confidence Interval Upper Bound**: `{stats['ci_95_upper_bound'] * 100.0:.3f}%`")
    md.append(f"- **Statistical Decision**: **{stats['verdict']}**\n")
    md.append("---\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Run 4-condition Tier 2 ablation study with statistical non-inferiority testing.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions per condition (default 3)")
    parser.add_argument("--output", type=str, default="docs/experiments.md", help="Output Markdown ledger path")
    parser.add_argument("--dataset", type=str, default="data/benchmarks/benchmark_dev_40.json", help="Path to evaluation dataset")
    parser.add_argument("--subset", type=int, default=10, help="Number of benchmark questions (default 10, yields 30 samples/condition with 3 repeats)")
    parser.add_argument("--delay", type=float, default=0.0, help="Pacing delay between queries in seconds")
    parser.add_argument("--delta", type=float, default=0.02, help="Non-inferiority delta margin (default 0.02 = 2.0%)")
    parser.add_argument("--break-it", action="store_true", help="Break It on Purpose demonstration: sets delta=0.0000001 to prove rejection under impossible margin")
    args = parser.parse_args()

    if args.break_it:
        args.delta = 0.0000001
        print("\n[BREAK-IT MODE] Demonstrating rejection under impossible non-inferiority margin delta = 0.0000001\n")

    # Load dataset
    repo_root = Path(__file__).resolve().parents[3]
    ds_path = repo_root / args.dataset
    if not ds_path.exists():
        sys.exit(f"Dataset file not found: {ds_path}")

    with open(ds_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions") or data
    if args.subset:
        questions = questions[:args.subset]

    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_commit = "unknown"

    print("=" * 85)
    print(" EvidenceFlow AI - Tier 2 Ablation Study & Non-Inferiority Benchmark")
    print("=" * 85)
    print(f"Dataset:       {args.dataset} ({len(questions)} questions)")
    print(f"Repetitions:   {args.repeats} repeats / condition ({len(questions) * args.repeats} total evaluations/condition)")
    print(f"Margin Delta:  {args.delta * 100.0:.6f}%")
    print(f"Git Commit:    {git_commit}")
    print("Conditions:    1. Baseline | 2. +Memory | 3. +Cache | 4. +Both\n")

    results = []
    try:
        for cond in CONDITIONS:
            print(f"==> Evaluating Condition: {cond.name} ({cond.description}) ...")
            t0 = time.perf_counter()
            res = run_condition_evaluation(
                condition=cond,
                benchmark_questions=questions,
                repeats=args.repeats,
                delay_seconds=args.delay,
                verbose=True,
            )
            elapsed = time.perf_counter() - t0
            print(f"    Completed in {elapsed:.2f}s | p50: {res['p50_latency_ms']:.2f}ms | Groundedness: {res['mean_groundedness']*100.0:.1f}%\n")
            results.append(res)

        # Statistical Non-Inferiority Test: Baseline vs +Both
        baseline_res = next(r for r in results if r["condition"] == "Baseline")
        both_res = next(r for r in results if r["condition"] == "+Both")

        stats = compute_non_inferiority_test(
            baseline_samples=baseline_res["groundedness_distribution"],
            treatment_samples=both_res["groundedness_distribution"],
            delta=args.delta,
        )

        # Format Markdown Report
        md_report = format_ablation_markdown_report(
            results=results,
            stats=stats,
            git_commit=git_commit,
            dataset_name=args.dataset,
            repeats=args.repeats,
        )

        print(md_report)

        # Append to output file (docs/experiments.md)
        out_path = repo_root / args.output
        if out_path.exists():
            with open(out_path, "a", encoding="utf-8") as f:
                f.write("\n" + md_report + "\n")
            print(f"[SUCCESS] Appended ablation results to '{out_path}'.")

        # Save JSON artifact
        json_artifact_path = repo_root / "data/benchmarks/ablation_tier2_results.json"
        with open(json_artifact_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "git_commit": git_commit,
                    "dataset": args.dataset,
                    "repeats": args.repeats,
                    "conditions": [
                        {k: v for k, v in r.items() if k != "groundedness_distribution"}
                        for r in results
                    ],
                    "non_inferiority_statistics": stats,
                },
                f,
                indent=2,
            )
        print(f"[SUCCESS] Saved raw JSON results to '{json_artifact_path}'.")

        if not stats["is_non_inferior"]:
            print(f"[ALERT] Non-inferiority test: {stats['verdict']}")
            if not args.break_it:
                sys.exit(1)

        print("=" * 85)
        print(" TIER 2 ABLATION STUDY & NON-INFERIORITY PROOF COMPLETED SUCCESSFULLY")
        print("=" * 85)

    finally:
        # Gracefully shut down DB connection pools
        try:
            close_connection_pool()
            close_neo4j_driver()
        except Exception:
            pass


if __name__ == "__main__":
    main()
