"""
run_followup_benchmark.py: Multi-turn conversational memory evaluation runner.
Evaluates 20 distinct multi-turn conversation scenarios in data/benchmarks/followup_eval_20.json.

Measures:
  - Resolution Accuracy (genuine pronoun follow-ups correctly mapped)
  - False Trigger Rate (explicit non-followup queries falsely rewritten)
  - Ambiguity Detection Rate (refusal / multi-entity turns correctly flagged as ambiguous)
  - Real Measured Execution Latency (mean, p50, p95, min, max in ms)
"""

import json
import uuid
import time
import statistics
import logging
from typing import Any, Dict, List
from evidenceflow.memory.init_neo4j import init_neo4j_schema, drop_graph
from evidenceflow.memory.graph_client import record_turn, resolve_entity_for_turn

logger = logging.getLogger("evidenceflow.eval.followup")


def run_followup_benchmark(dataset_path: str = "data/benchmarks/followup_eval_20.json") -> Dict[str, Any]:
    """
    Executes the 20-conversation multi-turn memory benchmark and computes empirical metrics.
    """
    init_neo4j_schema()

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    latencies = []

    for item in cases:
        case_id = item["id"]
        category = item["category"]
        turn1_spec = item.get("turn1")
        turn2_spec = item["turn2"]

        thread_id = f"bench-followup-{case_id}-{uuid.uuid4().hex[:6]}"
        drop_graph(thread_id=thread_id)

        # 1. Ingest Turn 1 into Neo4j memory if specified
        if turn1_spec is not None:
            turn1_id = f"{thread_id}_t1"
            record_turn(
                turn_id=turn1_id,
                thread_id=thread_id,
                turn_number=1,
                question=turn1_spec["question"],
                version_scope=turn1_spec.get("version"),
                entities=turn1_spec.get("entities", []),
                is_abstained=turn1_spec.get("is_abstained", False),
                abstention_code=turn1_spec.get("abstention_code"),
            )

        # 2. Measure Turn 2 resolution
        query = turn2_spec["query"]
        expected_status = turn2_spec["expected_status"]

        # Real timing measurement with high-resolution perf_counter
        t_start = time.perf_counter()
        resolution = resolve_entity_for_turn(thread_id=thread_id, follow_up_query=query)
        t_duration_ms = (time.perf_counter() - t_start) * 1000.0
        latencies.append(t_duration_ms)

        # 3. Evaluate correctness
        is_correct = False
        notes = []

        if expected_status == "resolved":
            expected_entity = turn2_spec["expected_entity"]
            expected_version = turn2_spec.get("expected_version")
            if resolution["status"] == "resolved":
                entity_match = resolution.get("entity_name") == expected_entity
                version_match = (expected_version is None) or (resolution.get("target_version") == expected_version)
                is_correct = entity_match and version_match
                if not entity_match:
                    notes.append(f"Entity mismatch: got {resolution.get('entity_name')}, expected {expected_entity}")
                if not version_match:
                    notes.append(f"Version mismatch: got {resolution.get('target_version')}, expected {expected_version}")
            else:
                notes.append(f"Expected 'resolved' but got '{resolution['status']}'")

        elif expected_status == "passthrough":
            if resolution["status"] == "passthrough":
                expected_entity = turn2_spec.get("expected_entity")
                if expected_entity:
                    is_correct = (resolution.get("entity_name") == expected_entity)
                else:
                    is_correct = True
            else:
                notes.append(f"False trigger: expected 'passthrough' but got '{resolution['status']}'")

        elif expected_status == "ambiguous":
            if resolution["status"] == "ambiguous":
                expected_reason = turn2_spec.get("expected_reason")
                if expected_reason:
                    is_correct = (resolution.get("reason") == expected_reason)
                else:
                    is_correct = True
            else:
                notes.append(f"Expected 'ambiguous' but got '{resolution['status']}'")

        results.append({
            "id": case_id,
            "category": category,
            "query": query,
            "expected_status": expected_status,
            "actual_status": resolution["status"],
            "is_correct": is_correct,
            "latency_ms": round(t_duration_ms, 3),
            "notes": "; ".join(notes) if notes else "PASS",
        })

        # Teardown thread nodes
        drop_graph(thread_id=thread_id)

    # 4. Metric Aggregations
    total = len(results)
    passed = sum(1 for r in results if r["is_correct"])
    overall_accuracy = (passed / total) * 100.0

    # Genuine pronoun follow-up subset (f01 - f08)
    pronoun_cases = [r for r in results if r["category"].startswith("single_entity_")]
    pronoun_correct = sum(1 for r in pronoun_cases if r["is_correct"])
    pronoun_accuracy = (pronoun_correct / len(pronoun_cases)) * 100.0 if pronoun_cases else 0.0

    # Explicit non-followup subset (f09 - f13)
    non_followup_cases = [r for r in results if r["category"].startswith("non_followup_")]
    false_triggers = sum(1 for r in non_followup_cases if r["actual_status"] != "passthrough")
    false_trigger_rate = (false_triggers / len(non_followup_cases)) * 100.0 if non_followup_cases else 0.0

    # Ambiguity detection subset (f14 - f20)
    ambiguity_cases = [r for r in results if r["expected_status"] == "ambiguous"]
    ambiguity_correct = sum(1 for r in ambiguity_cases if r["is_correct"])
    ambiguity_rate = (ambiguity_correct / len(ambiguity_cases)) * 100.0 if ambiguity_cases else 0.0

    sorted_lat = sorted(latencies)
    p50_latency = statistics.median(latencies)
    p95_index = int(0.95 * len(sorted_lat))
    p95_latency = sorted_lat[min(p95_index, len(sorted_lat) - 1)]

    summary = {
        "total_cases": total,
        "overall_accuracy_pct": round(overall_accuracy, 2),
        "pronoun_accuracy_pct": round(pronoun_accuracy, 2),
        "false_trigger_rate_pct": round(false_trigger_rate, 2),
        "ambiguity_detection_rate_pct": round(ambiguity_rate, 2),
        "latency_p50_ms": round(p50_latency, 3),
        "latency_p95_ms": round(p95_latency, 3),
        "latency_mean_ms": round(statistics.mean(latencies), 3),
        "latency_min_ms": round(min(latencies), 3),
        "latency_max_ms": round(max(latencies), 3),
        "details": results,
    }

    return summary


def print_benchmark_report(summary: Dict[str, Any]) -> None:
    """Prints a formatted evaluation report."""
    print("=" * 80)
    print(" EVIDENCEFLOW AI - MULTI-TURN GRAPH MEMORY BENCHMARK (20 CONVERSATIONS)")
    print("=" * 80)
    print(f" Total Evaluated Cases          : {summary['total_cases']}")
    print(f" Overall Resolution Accuracy     : {summary['overall_accuracy_pct']}% ({int(summary['total_cases'] * summary['overall_accuracy_pct'] / 100)} / {summary['total_cases']})")
    print(f" Pronoun Resolution Accuracy    : {summary['pronoun_accuracy_pct']}%")
    print(f" False Trigger Rate (Pass-through): {summary['false_trigger_rate_pct']}% (Target: 0.0%)")
    print(f" Ambiguity Detection Rate        : {summary['ambiguity_detection_rate_pct']}%")
    print("-" * 80)
    print(" REAL MEASURED EXECUTION LATENCIES (Python time.perf_counter()):")
    print(f"  * Median (p50)                 : {summary['latency_p50_ms']} ms")
    print(f"  * Tail (p95)                   : {summary['latency_p95_ms']} ms")
    print(f"  * Mean                         : {summary['latency_mean_ms']} ms")
    print(f"  * Min / Max                    : {summary['latency_min_ms']} ms / {summary['latency_max_ms']} ms")
    print("-" * 80)
    print(" DETAILED RESULTS TABLE:")
    print(f"{'ID':<5} | {'Category':<30} | {'Status':<12} | {'Latency':<9} | {'Outcome'}")
    print("-" * 80)
    for d in summary["details"]:
        status_str = f"{d['actual_status']}"
        outcome = "PASS" if d["is_correct"] else f"FAIL ({d['notes']})"
        print(f"{d['id']:<5} | {d['category']:<30} | {status_str:<12} | {d['latency_ms']:>6.2f} ms | {outcome}")
    print("=" * 80)


if __name__ == "__main__":
    summary = run_followup_benchmark()
    print_benchmark_report(summary)
