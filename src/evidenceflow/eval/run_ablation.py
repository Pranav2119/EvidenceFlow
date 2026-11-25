"""
run_ablation.py: Tier 2 Ablation Study Runner & Exploratory Quality Comparison.
Executes 4 pipeline configurations (Baseline, +Memory, +Cache, +Both) across benchmark splits,
recording pipeline self-checks and an approximate non-inferiority comparison.
Repeated executions are correlated; this calculation is not statistical proof.

CLI:
  uv run python -m evidenceflow.eval.run_ablation --repeats 3 --output docs/experiments.md
"""

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from evidenceflow.eval.ablation_runner import CONDITIONS, run_condition_evaluation
from evidenceflow.eval.metrics import get_dataset_metadata
from evidenceflow.memory.connection import close_neo4j_driver
from evidenceflow.storage.ledger import close_connection_pool


def compute_non_inferiority_test(
    baseline_samples: List[float],
    treatment_samples: List[float],
    delta: float = 0.02,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes an exploratory independent-sample normal approximation:
      H0: mu_baseline - mu_treatment >= delta (Treatment degrades quality by >= delta)
      H1: mu_baseline - mu_treatment < delta  (Treatment is non-inferior)

    Uses a one-sided (1 - alpha) upper confidence bound. Independent observations
    and an adequate normal approximation are assumptions, not verified here.
    Insufficient samples or zero estimated variance produce an inconclusive result.
    """
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    if not math.isfinite(delta) or not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be finite and between 0 and 1")
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for samples in (baseline_samples, treatment_samples)
        for value in samples
    ):
        raise ValueError("quality samples must be finite and between 0 and 1")

    n1 = len(baseline_samples)
    n2 = len(treatment_samples)
    m1 = sum(baseline_samples) / n1 if n1 else None
    m2 = sum(treatment_samples) / n2 if n2 else None
    diff = m1 - m2 if m1 is not None and m2 is not None else None
    result: Dict[str, Any] = {
        "mean_baseline": m1,
        "mean_treatment": m2,
        "mean_diff": diff,
        "standard_error": None,
        "ci_upper_bound": None,
        "confidence_level": 1.0 - alpha,
        "delta_margin": delta,
        "alpha": alpha,
        "z_stat": None,
        "p_value": None,
        "is_non_inferior": False,
        "verdict": "INCONCLUSIVE (At least two observations per condition are required)",
        "method": "Independent-sample normal approximation (exploratory)",
    }
    if n1 < 2 or n2 < 2 or m1 is None or m2 is None or diff is None:
        return result

    var1 = sum((x - m1) ** 2 for x in baseline_samples) / (n1 - 1)
    var2 = sum((x - m2) ** 2 for x in treatment_samples) / (n2 - 1)
    se = math.sqrt((var1 / n1) + (var2 / n2))
    result["standard_error"] = se
    if se <= 1e-9:
        result["verdict"] = "INCONCLUSIVE (Zero estimated variance cannot establish uncertainty)"
        return result

    z_crit = -NormalDist().inv_cdf(alpha)
    ci_upper = diff + (z_crit * se)
    z_stat = (delta - diff) / se
    p_val = NormalDist().cdf(-z_stat)

    is_non_inferior = ci_upper < delta
    verdict = (
        f"APPROXIMATE CHECK PASSED (Upper bound {ci_upper:.4f} < Margin {delta:.4f}, p={p_val:.4f})"
        if is_non_inferior
        else f"APPROXIMATE CHECK FAILED (Upper bound {ci_upper:.4f} >= Margin {delta:.4f}, p={p_val:.4f})"
    )

    result.update({
        "ci_upper_bound": ci_upper,
        "z_stat": z_stat,
        "p_value": p_val,
        "is_non_inferior": is_non_inferior,
        "verdict": verdict,
    })
    return result


def format_ablation_markdown_report(
    results: List[Dict[str, Any]],
    stats: Dict[str, Any],
    git_commit: str,
    dataset_name: str,
    repeats: int,
    dataset_metadata: Optional[Dict[str, Optional[str]]] = None,
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

    def format_stat(key: str, scale: float = 1.0, suffix: str = "") -> str:
        value = stats[key]
        return "N/A" if value is None else f"{value * scale:.4f}{suffix}"

    md = []
    md.append(f"\n## Milestone 9: Day 13 Tier 2 Ablation Study & Non-Inferiority Analysis\n")
    md.append(f"- **Git Commit**: `{git_commit}`")
    md.append(f"- **Dataset Split**: `{dataset_name}` ({results[0]['total_executions'] // repeats} questions x {repeats} repeats = {results[0]['total_executions']} runs/condition)")
    if dataset_metadata is not None:
        md.append(f"- **Dataset Version**: `{dataset_metadata['dataset_version']}`")
        md.append(f"- **Dataset SHA-256 (sorted compact JSON)**: `{dataset_metadata['dataset_sha256']}`")
    md.append(f"- **Non-Inferiority Margin ($\\Delta$)**: `{stats['delta_margin'] * 100.0:.3g} percentage points` (Significance Level $\\alpha = {stats['alpha']:g}$)")
    md.append(f"- **Evaluated Conditions**: Baseline, +Memory, +Cache, +Both (Full EvidenceFlow)\n")
    md.append(
        "**Exploratory results, not statistical proof.** Repeated executions of the same "
        "question are correlated, while this approximation assumes independent observations. "
        "The quality signal is the pipeline's L1 self-check (or an expected refusal), "
        "not independent evidence that every claim is supported. Token totals cover recorded "
        "audit steps; costs use the coded token-price estimates, not measured billing.\n"
    )

    md.append("### 1. Multi-Condition Performance & Cost Comparison Table\n")
    md.append("| Experimental Condition | Answer Checks (%) | Self-Check / Correct Refusal (%) | Median Latency (p50) | Tail Latency (p95) | Recorded Tokens / Query | Estimated Cost / 1k Queries | Cache Hit Rate |")
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
    if cache_only['cached_p50_latency_ms'] > 0 and baseline['p50_latency_ms'] > 0:
        cached_speedup = ((baseline['p50_latency_ms'] - cache_only['cached_p50_latency_ms']) / baseline['p50_latency_ms']) * 100.0
        md.append(f"- **Cached Pure Hit Speedup**: **{cached_speedup:.1f}%** reduction ({cache_only['cached_p50_latency_ms']:.2f}ms on verified cache hit)")
    md.append(f"- **Token Usage Reduction**: **{token_reduction:.1f}%** fewer tokens per query ({baseline['tokens_per_query']:.0f} $\\rightarrow$ {full_both['tokens_per_query']:.0f} tokens)")
    md.append(f"- **Token-Price Estimate Reduction**: **{cost_reduction:.1f}%** per 1,000 queries (not billed cost savings)")

    md.append("\n### 3. Exploratory Non-Inferiority Check\n")
    md.append(f"- **Null Hypothesis ($H_0$)**: $\\mu_{{\\text{{baseline}}}} - \\mu_{{\\text{{+Both}}}} \\ge {stats['delta_margin']:.6g}$")
    md.append(f"- **Alternative Hypothesis ($H_1$)**: $\\mu_{{\\text{{baseline}}}} - \\mu_{{\\text{{+Both}}}} < {stats['delta_margin']:.6g}$")
    md.append(f"- **Mean Self-Check Score (Baseline)**: `{format_stat('mean_baseline', 100.0, '%')}`")
    md.append(f"- **Mean Self-Check Score (+Both)**: `{format_stat('mean_treatment', 100.0, '%')}`")
    md.append(f"- **Mean Difference ($\\bar{{d}}$)**: `{format_stat('mean_diff', 100.0, ' percentage points')}`")
    md.append(f"- **Standard Error ($SE$)**: `{format_stat('standard_error')}`")
    md.append(f"- **Approximate {stats['confidence_level'] * 100.0:g}% Upper Confidence Bound**: `{format_stat('ci_upper_bound', 100.0, ' percentage points')}`")
    md.append(f"- **Exploratory Decision**: **{stats['verdict']}**\n")
    md.append("---\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Run 4-condition Tier 2 ablation study with an exploratory quality comparison.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions per condition (default 3)")
    parser.add_argument("--output", type=str, default="docs/experiments.md", help="Output Markdown ledger path")
    parser.add_argument("--dataset", type=str, default="data/benchmarks/benchmark_dev_40.json", help="Path to evaluation dataset")
    parser.add_argument("--subset", type=int, default=10, help="Number of benchmark questions (default 10; repeats are not independent samples)")
    parser.add_argument("--delay", type=float, default=0.0, help="Pacing delay between queries in seconds")
    parser.add_argument("--delta", type=float, default=0.02, help="Non-inferiority delta margin (default 0.02 = 2.0%)")
    parser.add_argument("--break-it", action="store_true", help="Use a strict delta=0.0000001 margin; this does not guarantee rejection or suppress a failure exit")
    args = parser.parse_args()

    if args.break_it:
        args.delta = 0.0000001
        print("\n[BREAK-IT MODE] Applying strict non-inferiority margin delta = 0.0000001\n")

    # Load dataset
    repo_root = Path(__file__).resolve().parents[3]
    ds_path = repo_root / args.dataset
    if not ds_path.exists():
        sys.exit(f"Dataset file not found: {ds_path}")

    with open(ds_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_metadata = get_dataset_metadata(data)
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
            dataset_metadata=dataset_metadata,
        )

        print(md_report)

        # Append to output file (docs/experiments.md)
        out_path = repo_root / args.output
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
                    **dataset_metadata,
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
            sys.exit(1)

        print("=" * 85)
        print(" TIER 2 ABLATION STUDY COMPLETED; EXPLORATORY CHECK PASSED (NOT STATISTICAL PROOF)")
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
