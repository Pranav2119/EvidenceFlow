"""
ablation_runner.py: Multi-condition ablation framework for EvidenceFlow AI.
Systematically evaluates 4 pipeline configurations:
  1. Baseline (Stateless, No Cache)
  2. +Memory (Neo4j Conversational Memory Graph only)
  3. +Cache (Scoped Cache enabled)
  4. +Both / Full System (Memory + Scoped Cache)

Measures latency reduction, token consumption, cost per 1k queries,
and collects sample distributions for statistical non-inferiority analysis.
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from evidenceflow.cache.manager import ScopedCacheManager
from evidenceflow.eval.metrics import (
    calculate_citation_precision,
    calculate_citation_recall,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    check_value_accuracy,
    is_wrong_answer,
    is_wrong_refusal,
)
from evidenceflow.graph.nodes.cache_gatekeeper import set_cache_manager
from evidenceflow.graph.workflow import run_pipeline
from evidenceflow.state import EvidenceFlowState

logger = logging.getLogger(__name__)

# Standard Gemini pricing (per million tokens) for cost estimation
GEMINI_FLASH_INPUT_COST_PER_M = 0.075   # $0.075 per 1M prompt tokens
GEMINI_FLASH_OUTPUT_COST_PER_M = 0.30   # $0.30 per 1M candidate tokens


class AblationCondition:
    """Descriptor for an experimental ablation condition."""
    def __init__(self, name: str, enable_memory: bool, enable_cache: bool, description: str):
        self.name = name
        self.enable_memory = enable_memory
        self.enable_cache = enable_cache
        self.description = description


CONDITIONS = [
    AblationCondition("Baseline", enable_memory=False, enable_cache=False, description="Stateless, No Cache"),
    AblationCondition("+Memory", enable_memory=True, enable_cache=False, description="Neo4j Memory Graph Active, No Cache"),
    AblationCondition("+Cache", enable_memory=False, enable_cache=True, description="Scoped Cache Enabled, No Memory"),
    AblationCondition("+Both", enable_memory=True, enable_cache=True, description="Full EvidenceFlow (Memory + Scoped Cache)"),
]


def calculate_cost_per_1k(total_prompt_tokens: int, total_completion_tokens: int, num_queries: int) -> float:
    """Calculates estimated LLM cost in USD per 1,000 queries."""
    if num_queries <= 0:
        return 0.0
    input_cost = (total_prompt_tokens / 1_000_000.0) * GEMINI_FLASH_INPUT_COST_PER_M
    output_cost = (total_completion_tokens / 1_000_000.0) * GEMINI_FLASH_OUTPUT_COST_PER_M
    total_cost = input_cost + output_cost
    return (total_cost / num_queries) * 1000.0


def calculate_percentiles(values: List[float], percentiles: List[float]) -> Dict[float, float]:
    """Computes exact percentiles for a list of floating-point measurements."""
    if not values:
        return {p: 0.0 for p in percentiles}
    s = sorted(values)
    res = {}
    for p in percentiles:
        k = (len(s) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            res[p] = s[int(k)]
        else:
            res[p] = s[int(f)] * (c - k) + s[int(c)] * (k - f)
    return res


def run_condition_evaluation(
    condition: AblationCondition,
    benchmark_questions: List[Dict[str, Any]],
    repeats: int = 3,
    delay_seconds: float = 0.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Executes a single ablation condition across benchmark questions repeated N times.
    """
    # Initialize fresh cache manager if cache is enabled, else clear it
    cm = ScopedCacheManager() if condition.enable_cache else None
    set_cache_manager(cm)

    latencies_all: List[float] = []
    latencies_cached_hits: List[float] = []
    groundedness_scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    cache_hits = 0
    total_executions = 0
    correct_count = 0

    num_q = len(benchmark_questions)

    for rep in range(1, repeats + 1):
        if verbose:
            print(f"   [{condition.name}] Running Repeat {rep}/{repeats} ({num_q} questions)...")

        for item in benchmark_questions:
            q_id = item.get("id") or item.get("question_id", "q")
            query = item["query"] if "query" in item else item.get("question", "")
            target_v = str(item.get("version") or item.get("target_version") or "16")

            if "should_abstain" in item:
                expected_answerable = not item["should_abstain"]
            else:
                expected_answerable = item.get("is_answerable", True)

            expected_values = item.get("expected_values")
            if not expected_values and item.get("expected_value") is not None:
                expected_values = [str(item["expected_value"])]
            elif not expected_values:
                expected_values = []

            gt_chunks = item.get("ground_truth_chunks", [])

            t0 = time.perf_counter()
            state: EvidenceFlowState = run_pipeline(
                query=query,
                version=target_v,
                thread_id=f"ablation_{condition.name}_rep{rep}_{q_id}",
                enable_memory=condition.enable_memory,
                enable_cache=condition.enable_cache,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            total_executions += 1
            latencies_all.append(elapsed_ms)

            is_hit = getattr(state, "cache_hit", False)
            if is_hit:
                cache_hits += 1
                latencies_cached_hits.append(elapsed_ms)

            # Extract token consumption from audit trail (0 on cache hits)
            if not is_hit:
                for step in getattr(state, "audit_trail", []):
                    total_prompt_tokens += step.get("prompt_tokens", 0) or 0
                    total_completion_tokens += step.get("candidate_tokens", 0) or 0

            # Compute quality metrics
            grounded = 1.0 if state.l1_passed and not state.is_abstained else 0.0
            if not expected_answerable and state.is_abstained:
                grounded = 1.0  # Clean abstention is correct
            groundedness_scores.append(grounded)

            retrieved_ids = {
                p.get("chunk_id") or p.get("id")
                for p in (state.retrieved_passages or [])
                if (p.get("chunk_id") or p.get("id"))
            }
            cits = state.citations or []
            p_prec = calculate_citation_precision(cits, retrieved_ids)
            p_rec = calculate_citation_recall(state.draft_answer, cits)
            precisions.append(p_prec)
            recalls.append(p_rec)

            # Check overall correctness
            if expected_answerable:
                val_ok = all(check_value_accuracy(state.draft_answer, value) for value in expected_values)
                if not state.is_abstained and state.l1_passed and val_ok:
                    correct_count += 1
            else:
                if state.is_abstained:
                    correct_count += 1

            if delay_seconds > 0.0:
                time.sleep(delay_seconds)

    pcts = calculate_percentiles(latencies_all, [50.0, 90.0, 95.0])
    cache_pcts = calculate_percentiles(latencies_cached_hits, [50.0, 95.0]) if latencies_cached_hits else {50.0: 0.0, 95.0: 0.0}

    mean_groundedness = (sum(groundedness_scores) / len(groundedness_scores)) if groundedness_scores else 0.0
    mean_precision = (sum(precisions) / len(precisions)) if precisions else 0.0
    mean_recall = (sum(recalls) / len(recalls)) if recalls else 0.0
    accuracy = (correct_count / total_executions) * 100.0 if total_executions > 0 else 0.0
    cache_hit_rate = (cache_hits / total_executions) * 100.0 if total_executions > 0 else 0.0

    cost_1k = calculate_cost_per_1k(total_prompt_tokens, total_completion_tokens, total_executions)

    return {
        "condition": condition.name,
        "description": condition.description,
        "enable_memory": condition.enable_memory,
        "enable_cache": condition.enable_cache,
        "repeats": repeats,
        "total_executions": total_executions,
        "accuracy_pct": accuracy,
        "mean_groundedness": mean_groundedness,
        "groundedness_distribution": groundedness_scores,
        "mean_citation_precision": mean_precision,
        "mean_citation_recall": mean_recall,
        "p50_latency_ms": pcts[50.0],
        "p95_latency_ms": pcts[95.0],
        "mean_latency_ms": sum(latencies_all) / len(latencies_all) if latencies_all else 0.0,
        "cached_p50_latency_ms": cache_pcts[50.0],
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "tokens_per_query": (total_prompt_tokens + total_completion_tokens) / total_executions if total_executions > 0 else 0.0,
        "cost_per_1k_usd": cost_1k,
        "cache_hits": cache_hits,
        "cache_hit_rate_pct": cache_hit_rate,
    }
